import torch
import syncode.common as common
from transformers import LogitsProcessor, PreTrainedTokenizer
from syncode.parse_result import AcceptSequence, RemainderState
from syncode.parsers.incremental_parser import IncrementalParser, ParseResult
from syncode.parsers import create_parser, create_base_parser
from syncode.dfa_mask_store import DFAMaskStore
from syncode.parsers.grammars import Grammar

# Set to True for debugging
DEBUG = False

class SyncodeLogitsProcessor2(LogitsProcessor):
    """
    A LogitsProcessor that filters model logits to only allow syntactically valid Python tokens.
    Now supports batch processing. Each element in the batch will be parsed independently.
    """

    def __init__(self, 
        grammar: Grammar, 
        tokenizer: PreTrainedTokenizer, 
        logger: common.Logger=common.EmptyLogger(), 
        use_cache=True,
        parse_output_only=True, 
        num_samples=1,
        dev_mode=False,
        parser='lalr',
        mode='grammar_mask'):

        self.tokenizer = tokenizer
        self.grammar = grammar
        self.logger = logger
        self.dev_mode = dev_mode
        self.num_samples = num_samples

        # We will store one parser and state per batch element
        self.inc_parsers = [None] * num_samples
        self.last_valid_state: list = [0] * num_samples
        self.function_ends: list = [None] * num_samples
        self.parse_output_only = parse_output_only
        self.start_from = [0] * num_samples

        self._ignore_whitespace = self._get_ignore_whitespace(self.grammar)

        # Load DFA mask store
        self.dfa_mask_store = DFAMaskStore.load_dfa_mask_store(
            grammar=self.grammar, 
            tokenizer=self.tokenizer, 
            use_cache=use_cache, 
            logger=self.logger,
            mode=mode
        )

        # We'll create parser instances per sample on reset.
        self.parser_type = parser

    def _log_current_status(self, partial_code, r: ParseResult):
        self.logger.log_code('Partial code', partial_code)
        self.logger.log(repr(r))

    def _get_ignore_whitespace(self, grammar):
        """
        Check if the grammar allows whitespace tokens to be ignored.
        """
        base_parser = create_base_parser(grammar)
        terminals = base_parser.terminals
        ignore_terminals = base_parser.ignore_tokens
        import regex
        ignore_whitespace = False
        for ig_name in ignore_terminals:
            for terminal in terminals:
                if terminal.name == ig_name:
                    if regex.match(terminal.pattern.to_regexp(), ' ') is not None:
                        ignore_whitespace = True
        return ignore_whitespace

    def reset(self, prompts):
        """
        Resets the decoder state on every new prompt or list of prompts.
        Args:
            prompts (str or list of str): The input prompt(s).
        """
        if isinstance(prompts, str):
            prompts = [prompts]  # Convert to list for uniform handling

        if len(prompts) != self.num_samples:
            raise ValueError("Number of prompts must match num_samples (batch size).")

        # Reset states for each sample
        self.last_valid_state = [0 for _ in range(self.num_samples)]
        self.function_ends = [None for _ in range(self.num_samples)]
        self.inc_parsers = [create_parser(self.grammar, logger=self.logger, parser=self.parser_type, ignore_whitespace=self._ignore_whitespace) 
                            for _ in range(self.num_samples)]

        for i, prompt in enumerate(prompts):
            prompt_tokens = self.tokenizer.encode(prompt, return_tensors='pt')[0]
            if self.parse_output_only:
                self.start_from[i] = len(prompt_tokens)
            else:
                self.start_from[i] = 0

            self.inc_parsers[i].reset()

    def is_valid(self, input_ids: torch.LongTensor, next_tokens: torch.LongTensor) -> torch.BoolTensor:
        """
        Check if the next tokens are valid for each sequence in the batch.

        Args:
            input_ids (torch.LongTensor): [batch_size, seq_len]
            next_tokens (torch.LongTensor): [batch_size]

        Returns:
            torch.BoolTensor: [batch_size] True/False validity per sequence.
        """
        batch_size = input_ids.shape[0]
        if batch_size != self.num_samples:
            raise ValueError("input_ids batch size must match num_samples.")

        validities = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        for idx in range(batch_size):
            inc_parser = self.inc_parsers[idx]
            # Append the next_token to the sequence
            new_input = torch.cat((input_ids[idx], next_tokens[idx].unsqueeze(0)), dim=0).unsqueeze(0)
            partial_code = self._get_partial_codes(new_input, idx)[0]

            try:
                r = inc_parser.get_acceptable_next_terminals(partial_code)
            except Exception as e:
                self.logger.log(f"Exception while parsing:\n {e}")
                validities[idx] = False
                continue

            # Check EOS handling
            if new_input[0, -1] == self.tokenizer.eos_token_id:
                # EOS is only allowed at $END or EOF
                validities[idx] = AcceptSequence(['$END']) in r.accept_sequences or AcceptSequence(['EOF']) in r.accept_sequences
            else:
                # Check if partial prefix is valid
                is_valid = self.dfa_mask_store.is_valid_prefix(r)
                if is_valid:
                    self.update_valid_state(partial_code, idx, r)
                validities[idx] = is_valid

        return validities

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:    
        # input_ids: [batch_size, seq_len]
        # scores: [batch_size, vocab_size]
        partial_codes_list = self._get_partial_codes(input_ids)

        for idx, partial_code in enumerate(partial_codes_list):
            inc_parser = self.inc_parsers[idx]

            # Parsing for this batch element
            try:
                r = inc_parser.get_acceptable_next_terminals(partial_code)
                self.update_valid_state(partial_code, idx, r)
            except Exception as e:
                if self.dev_mode:
                    raise e
                self.logger.log(f"Exception while parsing:\n {e}")
                continue

            accept_mask = self.dfa_mask_store.get_accept_mask(r, logger=self.logger)

            if DEBUG:
                self._log_current_status(partial_code, r)
                greedy_token = self.tokenizer.decode(scores[idx].argmax(dim=-1))

            if torch.sum(accept_mask) != 0:
                # Ensure accept_mask matches scores[idx] length
                if len(scores[idx]) != len(accept_mask):
                    diff = len(scores[idx]) - len(accept_mask)
                    if diff > 0:
                        accept_mask = torch.cat((accept_mask, torch.zeros(diff, dtype=torch.bool)))
                    else:
                        accept_mask = accept_mask[:len(scores[idx])]  # Truncate if needed

                scores[idx] = scores[idx].masked_fill(~accept_mask.to(scores.device), -float("inf"))
            else:
                # No acceptable tokens - log and leave scores as is (very restrictive scenario)
                self.logger.log('No acceptable tokens for the current partial code!')
                self._log_current_status(partial_code, r)

            # Debugging
            if DEBUG:
                self._debug_greedy(scores, idx, partial_code, r, greedy_token)

        return scores

    def _get_partial_codes(self, input_ids: torch.LongTensor, idx=None):
        """
        Decode the partial codes from input_ids. Each batch element might have a different start_from.
        If idx is None, we decode all batch elements.
        """
        if idx is not None:
            # Single index
            start = self.start_from[idx]
            partial_codes = self.tokenizer.batch_decode(input_ids[:, start:], skip_special_tokens=True)
        else:
            # Entire batch
            batch_partial_codes = []
            for i in range(input_ids.size(0)):
                start = self.start_from[i]
                code = self.tokenizer.decode(input_ids[i, start:], skip_special_tokens=True)
                batch_partial_codes.append(code)
            return batch_partial_codes
        return partial_codes

    def update_valid_state(self, partial_code: str, idx: int, r: ParseResult):
        """
        Update the last valid state and function ends for the given batch element.
        """
        if idx < len(self.function_ends):
            if r.function_end: 
                if self.function_ends[idx] is None:
                    self.function_ends[idx] = []
                self.function_ends[idx].append(len(partial_code) - len(r.remainder))

        if idx < len(self.last_valid_state):
            for accept_seq in r.accept_sequences:
                # 'EOF' or '$END' indicates the end
                if accept_seq[0] == '$END' or accept_seq[0] == 'EOF':
                    self.last_valid_state[idx] = len(partial_code) - len(r.remainder)

    def _debug_greedy(self, scores, idx, partial_code, r, greedy_token):
        greedy_grammar_token = self.tokenizer.decode(scores[idx].argmax(dim=-1))
        if greedy_token != greedy_grammar_token:
            self._log_greedy_difference(greedy_grammar_token, partial_code, r, greedy_token)

    def _log_greedy_difference(self, greedy_grammar_token, partial_code, r, greedy_token):
        self.logger.log_check("Greedy token and greedy grammar-based token do not match!")
        self.logger.log(f"Greedy token: {repr(greedy_token)}")
        self.logger.log(f"Greedy grammar-based token: {repr(greedy_grammar_token)}")
        self._log_current_status(partial_code, r)

    def print_debug(self):
        # Print debug info for the first parser (or all parsers)
        print('-'*50)
        print('Parsed terminals for the first parser:')
        if self.inc_parsers and self.inc_parsers[0]:
            name_to_pattern = {}
            for term in self.inc_parsers[0].base_parser.terminals:
                name_to_pattern[term.name] = term.pattern

            for token in self.inc_parsers[0].parsed_lexer_tokens:
                print(f"(type: {name_to_pattern[token.type]} | value: '{token.value}')")
        print('-'*50)

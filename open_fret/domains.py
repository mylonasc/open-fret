"""Declared instrument policy. Mixed-domain data is never disguised as guitar."""


def instrument_domain(program, strings):
    if program is not None and 24 <= program <= 31 and 6 <= strings <= 8:
        return 'guitar'
    if program is not None and 32 <= program <= 39:
        return 'bass'
    return 'other_or_unknown'


def accept_instrument(program, strings, policy='mixed'):
    if policy not in ('mixed', 'guitar'):
        raise ValueError('Expected mixed or guitar instrument policy')
    if not 1 <= strings <= 12:
        return False
    return policy == 'mixed' or instrument_domain(program, strings) == 'guitar'

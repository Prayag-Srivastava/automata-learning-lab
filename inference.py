"""
RPNI (Regular Positive and Negative Inference) learner.

Public API preserved for the application:
    build_pta(positive_strings, alphabet) -> DFA
    learn_dfa(positive_strings, negative_strings, alphabet) -> DFA

The learner uses a partial PTA internally. Missing transitions are implicit
rejections during inference; a complete "dead" sink is added only when the
final DFA is constructed.
"""

from collections import deque
from automaton import DFA


def _normalise_examples(positive_strings, negative_strings, alphabet):
    alphabet = set(alphabet)
    positives = set(positive_strings)
    negatives = set(negative_strings)

    for word in positives | negatives:
        if not isinstance(word, str):
            raise TypeError("Training examples must be strings.")
        invalid = set(word) - alphabet
        if invalid:
            raise ValueError(
                f"Example {word!r} contains symbols outside the alphabet: "
                f"{sorted(invalid)!r}"
            )

    conflicts = positives & negatives
    if conflicts:
        sample = sorted(conflicts)[:5]
        raise ValueError(
            "Contradictory training labels: these strings occur as both "
            f"positive and negative examples: {sample!r}"
        )

    return positives, negatives, alphabet


def _build_raw_pta(positive_strings, alphabet):
    """Build a partial prefix-tree acceptor using integer state IDs."""
    transitions = {0: {}}
    accepting = set()
    next_state = 1

    for word in sorted(set(positive_strings), key=lambda x: (len(x), x)):
        state = 0
        for symbol in word:
            if symbol not in alphabet:
                raise ValueError(
                    f"Positive example {word!r} contains symbol "
                    f"{symbol!r} outside the alphabet."
                )
            if symbol not in transitions[state]:
                transitions[state][symbol] = next_state
                transitions[next_state] = {}
                next_state += 1
            state = transitions[state][symbol]
        accepting.add(state)

    return transitions, accepting, 0


def _raw_to_dfa(transitions, accepting, start, alphabet):
    """Convert a partial integer-state graph into a complete DFA."""
    reachable = set()
    queue = deque([start])

    while queue:
        state = queue.popleft()
        if state in reachable:
            continue
        reachable.add(state)
        for symbol in alphabet:
            nxt = transitions.get(state, {}).get(symbol)
            if nxt is not None and nxt not in reachable:
                queue.append(nxt)

    # Stable names make graph rendering and debugging predictable.
    ordered = sorted(reachable)
    names = {state: f"q{state}" for state in ordered}
    dead = "dead"
    states = set(names.values()) | {dead}
    final_transitions = {}

    for state in ordered:
        src = names[state]
        for symbol in alphabet:
            nxt = transitions.get(state, {}).get(symbol)
            final_transitions[(src, symbol)] = names[nxt] if nxt is not None else dead

    for symbol in alphabet:
        final_transitions[(dead, symbol)] = dead

    final_accepting = {names[s] for s in accepting if s in reachable}
    return DFA(states, alphabet, final_transitions, names[start], final_accepting)


def build_pta(positive_strings, alphabet):
    """Public PTA builder used by the UI."""
    positives, _, alphabet = _normalise_examples(
        positive_strings, (), alphabet
    )
    transitions, accepting, start = _build_raw_pta(positives, alphabet)
    return _raw_to_dfa(transitions, accepting, start, alphabet)


def _make_parent(states):
    return {state: state for state in states}


def _find(parent, state):
    while parent[state] != state:
        parent[state] = parent[parent[state]]
        state = parent[state]
    return state


def _members(parent):
    result = {}
    for state in parent:
        result.setdefault(_find(parent, state), []).append(state)
    return result


def _merge_with_folding(pta_transitions, pta_accepting, parent,
                        state_a, state_b, alphabet):
    """
    Tentatively merge two equivalence classes.

    If both classes have a transition on the same symbol, their destination
    classes must also be merged. This recursive propagation is RPNI folding.
    Returns a new parent map. Accepting and non-accepting PTA states may be
    merged: a non-accepting PTA state is merely *unlabelled*, not known to be
    negative. Consistency with the negatives is checked by the caller.
    """
    trial = parent.copy()
    pending = [(state_a, state_b)]

    while pending:
        a, b = pending.pop()
        ra = _find(trial, a)
        rb = _find(trial, b)

        if ra == rb:
            continue

        members = _members(trial)
        group_a = members[ra]
        group_b = members[rb]

        # Deterministic representative selection; preserve the root if involved.
        if 0 in group_a:
            keep, drop = ra, rb
        elif 0 in group_b:
            keep, drop = rb, ra
        else:
            keep = min(min(group_a), min(group_b))
            drop = rb if keep in group_a else ra

        # Redirect every member of the dropped class to the kept class.
        for state in group_a + group_b:
            trial[state] = keep

        # Fold successor classes for symbols present on both sides.
        # A transition present on only one side is retained in the quotient.
        representative_members = group_a + group_b
        for symbol in alphabet:
            destinations = set()
            for state in representative_members:
                nxt = pta_transitions.get(state, {}).get(symbol)
                if nxt is not None:
                    destinations.add(_find(trial, nxt))
            if len(destinations) > 1:
                first = min(destinations)
                for other in sorted(destinations):
                    if other != first:
                        pending.append((first, other))

    return trial


def _quotient_is_consistent(parent, pta_transitions, pta_accepting,
                            negative_strings, start, alphabet):
    """Check that the quotient rejects every supplied negative example."""
    members = _members(parent)
    class_accepting = {
        root: any(state in pta_accepting for state in group)
        for root, group in members.items()
    }

    # Build deterministic quotient transitions.
    quotient_transitions = {}
    for state, edges in pta_transitions.items():
        src = _find(parent, state)
        for symbol, destination in edges.items():
            dst = _find(parent, destination)
            key = (src, symbol)
            previous = quotient_transitions.get(key)
            if previous is not None and previous != dst:
                # A valid folded quotient should have merged these targets.
                return False
            quotient_transitions[key] = dst

    root_start = _find(parent, start)
    for word in negative_strings:
        state = root_start
        rejected_early = False
        for symbol in word:
            nxt = quotient_transitions.get((state, symbol))
            if nxt is None:
                rejected_early = True
                break
            state = nxt
        if not rejected_early and class_accepting.get(state, False):
            return False

    return True


def _reachable_quotient(parent, pta_transitions, start, alphabet):
    """Return reachable quotient classes and their partial transitions."""
    members = _members(parent)
    start_root = _find(parent, start)
    queue = deque([start_root])
    seen = set()
    quotient_transitions = {}

    while queue:
        root = _find(parent, queue.popleft())
        if root in seen:
            continue
        seen.add(root)

        for symbol in alphabet:
            destinations = {
                _find(parent, pta_transitions[state][symbol])
                for state in members[root]
                if symbol in pta_transitions.get(state, {})
            }
            if not destinations:
                continue
            # Folding should guarantee one destination per symbol.
            if len(destinations) != 1:
                raise RuntimeError("Internal RPNI error: quotient is nondeterministic.")
            destination = next(iter(destinations))
            quotient_transitions[(root, symbol)] = destination
            if destination not in seen:
                queue.append(destination)

    return seen, quotient_transitions, start_root


def _minimize_complete_dfa(states, alphabet, transitions, start, accepting):
    """Partition-refinement minimization; preserves the recognized language."""
    states = set(states)
    accepting = set(accepting)
    rejecting = states - accepting
    partitions = [part for part in (accepting, rejecting) if part]

    while True:
        block_of = {
            state: index
            for index, block in enumerate(partitions)
            for state in block
        }
        refined = []

        for block in partitions:
            buckets = {}
            for state in block:
                signature = tuple(
                    block_of[transitions[(state, symbol)]]
                    for symbol in sorted(alphabet)
                )
                buckets.setdefault(signature, set()).add(state)
            refined.extend(buckets.values())

        if len(refined) == len(partitions):
            partitions = refined
            break
        partitions = refined

    block_of = {
        state: index
        for index, block in enumerate(partitions)
        for state in block
    }
    # BFS numbering from the start block.
    start_block = block_of[start]
    order = [start_block]
    queue = deque([start_block])
    seen = {start_block}

    while queue:
        block_index = queue.popleft()
        representative = next(iter(partitions[block_index]))
        for symbol in sorted(alphabet):
            destination_block = block_of[transitions[(representative, symbol)]]
            if destination_block not in seen:
                seen.add(destination_block)
                order.append(destination_block)
                queue.append(destination_block)

    names = {block_index: f"q{i}" for i, block_index in enumerate(order)}
    minimized_states = set(names.values())
    minimized_transitions = {}
    minimized_accepting = set()

    for block_index in order:
        representative = next(iter(partitions[block_index]))
        source_name = names[block_index]
        if representative in accepting:
            minimized_accepting.add(source_name)
        for symbol in alphabet:
            destination_block = block_of[transitions[(representative, symbol)]]
            minimized_transitions[(source_name, symbol)] = names[destination_block]

    return DFA(
        minimized_states, alphabet, minimized_transitions,
        names[start_block], minimized_accepting
    )


def learn_dfa(positive_strings, negative_strings, alphabet):
    """
    Learn a DFA using standard RPNI red-blue state merging.

    RPNI uses only the supplied positive and negative examples. It does not
    inspect or use a reference DFA. The returned DFA is minimized afterward.
    """
    positives, negatives, alphabet = _normalise_examples(
        positive_strings, negative_strings, alphabet
    )
    pta_transitions, pta_accepting, start = _build_raw_pta(
        positives, alphabet
    )

    # RPNI assumes the sample is consistent. Check this explicitly.
    if not _quotient_is_consistent(
        {s: s for s in pta_transitions},
        pta_transitions, pta_accepting, negatives, start, alphabet
    ):
        raise ValueError(
            "The training sample is inconsistent: the positive PTA already "
            "accepts at least one negative example."
        )

    parent = _make_parent(pta_transitions)
    red = {start}

    # Stable PTA breadth-first order.
    bfs_order = []
    queue = deque([start])
    seen = set()
    while queue:
        state = queue.popleft()
        if state in seen:
            continue
        seen.add(state)
        bfs_order.append(state)
        for symbol in sorted(alphabet):
            nxt = pta_transitions.get(state, {}).get(symbol)
            if nxt is not None and nxt not in seen:
                queue.append(nxt)

    # Red-blue RPNI: process the frontier in PTA BFS order.
    while True:
        members = _members(parent)
        red_roots = {_find(parent, state) for state in red}

        blue_roots = set()
        for red_root in red_roots:
            for member in members.get(red_root, []):
                for symbol in alphabet:
                    nxt = pta_transitions.get(member, {}).get(symbol)
                    if nxt is not None:
                        nxt_root = _find(parent, nxt)
                        if nxt_root not in red_roots:
                            blue_roots.add(nxt_root)

        if not blue_roots:
            break

        blue = next(
            _find(parent, state)
            for state in bfs_order
            if _find(parent, state) in blue_roots
        )

        merged = False
        for red_state in sorted(red_roots, key=lambda x: (
            0 if x == _find(parent, start) else 1, x
        )):
            candidate = _merge_with_folding(
                pta_transitions, pta_accepting, parent,
                red_state, blue, alphabet
            )
            if candidate is None:
                continue
            if _quotient_is_consistent(
                candidate, pta_transitions, pta_accepting,
                negatives, start, alphabet
            ):
                parent = candidate
                merged = True
                break

        if not merged:
            # No compatible red state: promote this blue state to red.
            red.add(blue)

    # Construct the reachable partial quotient.
    quotient_states, quotient_transitions, quotient_start = _reachable_quotient(
        parent, pta_transitions, start, alphabet
    )
    members = _members(parent)
    quotient_accepting = {
        root for root in quotient_states
        if any(state in pta_accepting for state in members[root])
    }

    # Complete the DFA with one rejecting sink for missing transitions.
    dead = "dead"
    used_names = {f"q{root}" for root in quotient_states}
    if dead in used_names:
        dead = "__dead__"

    names = {root: f"q{root}" for root in quotient_states}
    complete_states = set(names.values()) | {dead}
    complete_transitions = {}
    complete_accepting = {names[root] for root in quotient_accepting}

    for root in quotient_states:
        source = names[root]
        for symbol in alphabet:
            destination = quotient_transitions.get((root, symbol))
            complete_transitions[(source, symbol)] = (
                names[destination] if destination is not None else dead
            )

    for symbol in alphabet:
        complete_transitions[(dead, symbol)] = dead

    complete_start = names[quotient_start]

    # Minimize the learned DFA; this is a language-preserving post-processing
    # step, not access to the reference automaton.
    return _minimize_complete_dfa(
        complete_states, alphabet, complete_transitions,
        complete_start, complete_accepting
    )

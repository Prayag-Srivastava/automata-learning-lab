class DFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states=set(states); self.alphabet=set(alphabet); self.transitions=dict(transitions)
        self.start_state=start_state; self.accept_states=set(accept_states)
    def _step(self, state, symbol):
        return self.transitions.get((state,symbol), 'dead')
    def accepts(self, string):
        state=self.start_state
        for c in string:
            if c not in self.alphabet: return False
            state=self._step(state,c)
        return state in self.accept_states
    def trace(self, string):
        state=self.start_state; out=[]
        for c in string:
            nxt=self._step(state,c); out.append((state,c,nxt)); state=nxt
        return out
    def num_states(self): return len(self.states)
    def is_equivalent_to(self, other_dfa, test_strings):
        xs=list(test_strings)
        return sum(self.accepts(x)==other_dfa.accepts(x) for x in xs)/len(xs) if xs else 1.0

class NFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states=set(states); self.alphabet=set(alphabet); self.transitions={k:set(v) for k,v in transitions.items()}
        self.start_state=start_state; self.accept_states=set(accept_states)
    def _closure(self, states):
        result=set(states); stack=list(states)
        while stack:
            q=stack.pop()
            for nxt in self.transitions.get((q,'ε'),set())-result: result.add(nxt); stack.append(nxt)
        return result
    def accepts(self, string):
        cur=self._closure({self.start_state})
        for c in string:
            cur=self._closure(set().union(*(self.transitions.get((q,c),set()) for q in cur))) if cur else set()
        return bool(cur & self.accept_states)
    def trace(self,string):
        d=self.to_dfa(); return d.trace(string)
    def to_dfa(self):
        start=frozenset(self._closure({self.start_state})); queue=[start]; seen={start}; trans={}; names={start:'q0'}
        while queue:
            subset=queue.pop(0); q=names[subset]
            for c in sorted(self.alphabet):
                nxt=frozenset(self._closure(set().union(*(self.transitions.get((s,c),set()) for s in subset))))
                if not nxt: continue
                if nxt not in seen: seen.add(nxt); queue.append(nxt); names[nxt]=f'q{len(names)}'
                trans[(q,c)]=names[nxt]
        accepts={names[s] for s in seen if s & self.accept_states}
        return DFA(set(names.values()),self.alphabet,trans,'q0',accepts)

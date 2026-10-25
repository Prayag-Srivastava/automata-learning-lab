from automaton import DFA

def build_pta(positive_strings, alphabet):
    transitions={}; states={'q0'}; accepting=set(); prefix_state={'': 'q0'}; counter=1
    for word in sorted(set(positive_strings)):
        state='q0'; prefix=''
        if not word: accepting.add('q0')
        for c in word:
            if c not in alphabet: continue
            prefix+=c
            if (state,c) not in transitions:
                nxt=f'q{counter}'; counter+=1; states.add(nxt); transitions[(state,c)]=nxt; prefix_state[prefix]=nxt
            state=transitions[(state,c)]
        accepting.add(state)
    states.add('dead')
    for q in list(states):
        if q!='dead':
            for c in alphabet: transitions.setdefault((q,c),'dead')
    for c in alphabet: transitions[('dead',c)]='dead'
    return DFA(states,alphabet,transitions,'q0',accepting)

def _quotient(pt, groups):
    root=lambda x: groups.get(x,x)
    # resolve chains
    def find(x):
        while root(x)!=x: x=root(x)
        return x
    trans={}
    for (q,c),n in pt.transitions.items(): trans[(find(q),c)]=find(n)
    states={find(q) for q in pt.states}; accepts={find(q) for q in pt.accept_states}
    return DFA(states,pt.alphabet,trans,find(pt.start_state),accepts)

def learn_dfa(positive_strings, negative_strings, alphabet):
    pt=build_pta(positive_strings,alphabet); groups={}; states=sorted(pt.states, key=lambda q:(q!='q0', int(q[1:]) if q.startswith('q') and q[1:].isdigit() else 10**9))
    # Greedy deterministic state merging. Each trial closes the merge under transitions.
    parent={q:q for q in pt.states}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def trial(a,b):
        p=parent.copy(); todo=[(find(a),find(b))]
        while todo:
            x,y=todo.pop()
            rx,ry=x,y
            while p[rx]!=rx: rx=p[rx]
            while p[ry]!=ry: ry=p[ry]
            if rx==ry: continue
            # Avoid merging dead into a live accepting state.
            if (rx=='dead' and ry in pt.accept_states) or (ry=='dead' and rx in pt.accept_states): return None
            if (rx in pt.accept_states) != (ry in pt.accept_states): return None
            keep,drop=sorted([rx,ry],key=lambda q:(q!='q0', q=='dead', q))
            p[drop]=keep
            for c in sorted(alphabet):
                nx=pt.transitions.get((rx,c),'dead'); ny=pt.transitions.get((ry,c),'dead')
                todo.append((nx,ny))
        return p
    for i,a in enumerate(states):
        for b in states[i+1:]:
            ra,rb=find(a),find(b)
            if ra==rb: continue
            # Dead remains distinct from accepting states, including by explicit rule.
            candidate=trial(ra,rb)
            if candidate is None: continue
            d=_quotient(pt,candidate)
            if any(d.accepts(w) for w in negative_strings): continue
            parent=candidate
            for q in list(parent): find(q)
    d=_quotient(pt,{q:find(q) for q in pt.states})
    # Prune unreachable states; transitions remain total for reachable states.
    reachable={d.start_state}; todo=[d.start_state]
    while todo:
        q=todo.pop()
        for c in alphabet:
            n=d.transitions.get((q,c),'dead')
            if n not in reachable: reachable.add(n); todo.append(n)
    return DFA(reachable,alphabet,{k:v for k,v in d.transitions.items() if k[0] in reachable and v in reachable},d.start_state,d.accept_states & reachable)

import random, re, itertools

# Hand-coded complete minimal DFAs. Missing transitions are handled as a rejecting sink.
LANGUAGES = {
    "Even number of a's": dict(name="Even number of a's", alphabet=["a","b"], regex_pattern=r"^(b*ab*ab*)*b*$", reference_dfa={"states":["q0","q1"],"start_state":"q0","accept_states":["q0"],"transitions":{("q0","a"):"q1",("q0","b"):"q0",("q1","a"):"q0",("q1","b"):"q1"}}),
    "Ends with 'ab'": dict(name="Ends with 'ab'", alphabet=["a","b"], regex_pattern=r"^(a|b)*ab$", reference_dfa={"states":["q0","q1","q2"],"start_state":"q0","accept_states":["q2"],"transitions":{("q0","a"):"q1",("q0","b"):"q0",("q1","a"):"""q1""",("q1","b"):"q2",("q2","a"):"q1",("q2","b"):"q0"}}),
    "Contains substring 'aba'": dict(name="Contains substring 'aba'", alphabet=["a","b"], regex_pattern=r"^(a|b)*aba(a|b)*$", reference_dfa={"states":["q0","q1","q2","q3"],"start_state":"q0","accept_states":["q3"],"transitions":{("q0","a"):"q1",("q0","b"):"q0",("q1","a"):"q1",("q1","b"):"q2",("q2","a"):"q3",("q2","b"):"q0",("q3","a"):"q3",("q3","b"):"q3"}}),
    "Strings of the form a*b*": dict(name="Strings of the form a*b*", alphabet=["a","b"], regex_pattern=r"^a*b*$", reference_dfa={"states":["q0","q1","q2"],"start_state":"q0","accept_states":["q0","q1"],"transitions":{("q0","a"):"q0",("q0","b"):"q1",("q1","b"):"q1",("q1","a"):"q2",("q2","a"):"q2",("q2","b"):"q2"}}),
    "Length divisible by 3": dict(name="Length divisible by 3", alphabet=["a","b"], regex_pattern=r"^((a|b)(a|b)(a|b))*$", reference_dfa={"states":["q0","q1","q2"],"start_state":"q0","accept_states":["q0"],"transitions":{(f"q{i}",c):f"q{(i+1)%3}" for i in range(3) for c in "ab"}}),
    "No two consecutive a's": dict(name="No two consecutive a's", alphabet=["a","b"], regex_pattern=r"^(b|ab)*a?$", reference_dfa={"states":["q0","q1","q2"],"start_state":"q0","accept_states":["q0","q1"],"transitions":{("q0","a"):"q1",("q0","b"):"q0",("q1","a"):"q2",("q1","b"):"q0",("q2","a"):"q2",("q2","b"):"q2"}})
}

def generate_samples(language_name, n_positive, n_negative, max_length, noise_fraction=0.0):
    lang=LANGUAGES[language_name]; alphabet=lang['alphabet']; pat=re.compile(lang['regex_pattern'])
    pos=[]; neg=[]; seen=set(); max_space=sum(len(alphabet)**i for i in range(max_length+1))
    # Randomized sampling followed by exhaustive enumeration guarantees finite spaces terminate.
    for _ in range(max(1000, min(max_space*3, 100000))):
        if len(pos)>=n_positive and len(neg)>=n_negative: break
        s=''.join(random.choices(alphabet,k=random.randint(0,max_length)))
        if s in seen: continue
        seen.add(s); (pos if pat.fullmatch(s) else neg).append(s)
    if len(pos)<n_positive or len(neg)<n_negative:
        for n in range(max_length+1):
            for chars in itertools.product(alphabet, repeat=n):
                s=''.join(chars)
                if s not in seen:
                    seen.add(s); (pos if pat.fullmatch(s) else neg).append(s)
                if len(pos)>=n_positive and len(neg)>=n_negative: break
            if len(pos)>=n_positive and len(neg)>=n_negative: break
    # Sampling with replacement is appropriate when a finite language has fewer
    # distinct strings than the requested dataset size (e.g. a*b* at length 10).
    if len(pos)<n_positive and pos:
        pos += random.choices(pos, k=n_positive-len(pos))
    if len(neg)<n_negative and neg:
        neg += random.choices(neg, k=n_negative-len(neg))
    if len(pos)<n_positive or len(neg)<n_negative:
        raise ValueError("No examples of one label exist within this maximum string length. Increase max length.")
    samples=[(s,1) for s in pos[:n_positive]]+[(s,0) for s in neg[:n_negative]]
    random.shuffle(samples)
    flips=round(len(samples)*noise_fraction)
    # Flip labels per *string*, not per sample slot: when duplicates exist (see
    # above), flipping only one copy would give the same string both labels,
    # which no DFA can represent and which learn_dfa rejects as contradictory.
    flipped={samples[i][0] for i in random.sample(range(len(samples)), min(flips,len(samples)))}
    samples=[(s,1-y) if s in flipped else (s,y) for s,y in samples]
    return [s for s,_ in samples],[y for _,y in samples]

def get_reference_dfa(language_name):
    d=LANGUAGES[language_name]['reference_dfa']
    return {"states":list(d['states']),"start_state":d['start_state'],"accept_states":list(d['accept_states']),"transitions":dict(d['transitions'])}

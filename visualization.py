import graphviz

def render_dfa(dfa, highlight_path=None, highlight_state=None):
    dot=graphviz.Digraph(engine='dot', graph_attr={'rankdir':'LR','bgcolor':'white','pad':'0.25'}, node_attr={'fontname':'Arial','fontsize':'11'}, edge_attr={'fontname':'Arial','fontsize':'10'})
    dot.attr('node',shape='point'); dot.node('__start'); dot.edge('__start',dfa.start_state)
    dot.attr('node',shape='circle')
    path=set(highlight_path or []); edge_pairs=set(zip(highlight_path or [],(highlight_path or [])[1:]))
    for q in sorted(dfa.states):
        attrs={'style':'filled' if q in path or q==highlight_state else 'solid','fillcolor':'#dcfce7' if q in path else ('#dbeafe' if q==highlight_state else 'white'),'color':'#16a34a' if q in path else ('#2563eb' if q==highlight_state else '#334155'),'penwidth':'2' if q in path or q==highlight_state else '1'}
        dot.node(q,shape='doublecircle' if q in dfa.accept_states else 'circle',**attrs)
    grouped={}
    for (q,c),n in dfa.transitions.items(): grouped.setdefault((q,n),[]).append(c)
    for (q,n),symbols in sorted(grouped.items()):
        color='#16a34a' if (q in path and n in path and (q,n) in edge_pairs) else '#64748b'
        dot.edge(q,n,label=','.join(sorted(symbols)),color=color,fontcolor=color,penwidth='2' if color!='#64748b' else '1')
    return dot

def render_pta_vs_learned(pta, learned): return render_dfa(pta),render_dfa(learned)

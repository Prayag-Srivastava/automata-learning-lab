import random, itertools
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # fixes your earlier crashfrom sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from dataset_generator import LANGUAGES, generate_samples, get_reference_dfa
from automaton import DFA,NFA
from inference import build_pta, learn_dfa
from ml_classifier import extract_features, train_and_evaluate
from visualization import render_dfa

st.set_page_config(page_title='Automata Learning Lab',layout='wide',page_icon='🤖')
st.markdown('''<style>.main{background:#f7f9fc}.block-container{padding-top:1.4rem}.hero{padding:1.2rem 1.5rem;background:linear-gradient(110deg,#101e38,#263e64);border-radius:18px;color:white;margin-bottom:1rem}.hero h1{margin:0;color:white}.metric-card{padding:1rem;border:1px solid #e5eaf2;border-radius:14px;background:white}</style>''',unsafe_allow_html=True)

@st.cache_data
def cached_samples(language,npos,nneg,length,noise,seed):
    random.seed(seed); return generate_samples(language,npos,nneg,length,noise)

def make_dfa(data, alphabet):
    return DFA(data['states'],alphabet,data['transitions'],data['start_state'],data['accept_states'])
def random_words(alphabet,max_len,n,seed=711):
    r=random.Random(seed); return [''.join(r.choices(alphabet,k=r.randint(0,max_len))) for _ in range(n)]
def metrics(y,p):
    return {'Accuracy':accuracy_score(y,p),'Precision':precision_score(y,p,zero_division=0),'Recall':recall_score(y,p,zero_division=0),'F1-Score':f1_score(y,p,zero_division=0),'confusion_matrix':confusion_matrix(y,p,labels=[0,1])}

with st.sidebar:
    st.title('🤖 Automata Learning Lab')
    language=st.selectbox('Choose Language',list(LANGUAGES))
    npos=st.slider('Number of positive examples',10,500,100)
    nneg=st.slider('Number of negative examples',10,500,100)
    maxlen=st.slider('Max string length',3,20,10)
    noise=st.slider('Noise fraction',0.0,0.3,0.0,0.05)
    if 'seed' not in st.session_state: st.session_state.seed=123
    if st.button('🔄 Generate New Dataset',use_container_width=True): st.session_state.seed+=1
    st.divider(); st.markdown('**About**\n\nA TOC course project combining formal language theory with machine learning. Explore regular languages, infer automata from examples, and compare symbolic learning with classifiers.')

alphabet=LANGUAGES[language]['alphabet']
try:
    strings,labels=cached_samples(language,npos,nneg,maxlen,noise,st.session_state.seed)
except ValueError as e:
    st.error(str(e)); st.stop()
ref=make_dfa(get_reference_dfa(language),alphabet)
if 'learned_key' not in st.session_state: st.session_state.learned_key=None

def current_learned():
    return st.session_state.get('learned_dfa') if st.session_state.get('learned_key')==(language,st.session_state.seed) else None

st.markdown('<div class="hero"><h1>Automata Learning Lab</h1><div>From labeled examples to learned machines — inspect, test, and compare.</div></div>',unsafe_allow_html=True)
tabs=st.tabs(['📊 Dataset','🧠 Learn Automaton','🔍 Test Strings','📈 ML Comparison','🧪 Experiments'])
with tabs[0]:
    c1,c2,c3=st.columns(3)
    c1.metric('Examples',len(strings)); c2.metric('Positive',sum(labels)); c3.metric('Negative',len(labels)-sum(labels))
    df=pd.DataFrame({'String':strings,'Length':[len(s) for s in strings],'Label':['Accept' if y else 'Reject' for y in labels]})
    st.dataframe(df,use_container_width=True,height=350)
    st.subheader('Ground-Truth Minimal DFA'); st.graphviz_chart(render_dfa(ref),use_container_width=True)
    st.caption(f"Reference machine for **{language}** · {ref.num_states()} states · regex `{LANGUAGES[language]['regex_pattern']}`")

with tabs[1]:
    if st.button('▶ Run Learning Algorithm',type='primary'):
        pos=[s for s,y in zip(strings,labels) if y==1]; neg=[s for s,y in zip(strings,labels) if y==0]
        with st.spinner('Learning automaton...'):
            pta=build_pta(pos,alphabet); learned=learn_dfa(pos,neg,alphabet)
        st.session_state.pta=pta; st.session_state.learned_dfa=learned; st.session_state.learned_key=(language,st.session_state.seed)
    learned=current_learned()
    if learned:
        pos=[s for s,y in zip(strings,labels) if y==1]; pta=st.session_state.pta
        cols=st.columns(3)
        with cols[0]: st.caption(f'Step 1: Prefix Tree Acceptor — {pta.num_states()} states'); st.graphviz_chart(render_dfa(pta),use_container_width=True)
        with cols[1]: st.caption(f'Step 2: After State Merging — {learned.num_states()} states'); st.graphviz_chart(render_dfa(learned),use_container_width=True)
        with cols[2]: st.caption(f'Reference Minimal DFA — {ref.num_states()} states'); st.graphviz_chart(render_dfa(ref),use_container_width=True)
        tests=random_words(alphabet,maxlen,1000); equiv=learned.is_equivalent_to(ref,tests)
        order=list(range(len(strings))); random.Random(42).shuffle(order); cut=max(1,int(.8*len(order))); testids=order[cut:]
        acc=sum(learned.accepts(strings[i])==labels[i] for i in testids)/max(len(testids),1)
        a,b,c=st.columns(3); a.metric('Learned / reference states',f'{learned.num_states()} / {ref.num_states()}'); b.metric('Approx. equivalence',f'{equiv:.1%}'); c.metric('Held-out accuracy',f'{acc:.1%}')
    else: st.info('Click **Run Learning Algorithm** to build a PTA and merge compatible states.')

with tabs[2]:
    learned=current_learned()
    if not learned: st.info('Learn an automaton in the previous tab first.')
    else:
        value=st.text_input('Enter a string to test',key='single_test')
        if st.button('▶ Run on Learned DFA',type='primary'):
            ok=learned.accepts(value); st.markdown(f"<h2 style='color:{'#15803d' if ok else '#b91c1c'}'>{'✅ ACCEPTED' if ok else '❌ REJECTED'}</h2>",unsafe_allow_html=True)
            trace=learned.trace(value); states=[learned.start_state]+[t[2] for t in trace]
            table=[{'Step':i+1,'Current State':q,'Symbol Read':c,'Next State':n} for i,(q,c,n) in enumerate(trace)]
            st.dataframe(pd.DataFrame(table,columns=['Step','Current State','Symbol Read','Next State']),use_container_width=True)
            st.graphviz_chart(render_dfa(learned,highlight_path=states),use_container_width=True)
            truth=ref.accepts(value); st.info(f"Reference DFA: **{'ACCEPTED' if truth else 'REJECTED'}** — {'agrees' if truth==ok else 'disagrees'} with the learned DFA.")
        with st.expander('Batch Test'):
            batch=st.text_area('Paste one string per line',key='batch')
            if st.button('Test batch'):
                rows=[]
                for s in batch.splitlines():
                    pred=learned.accepts(s); truth=ref.accepts(s); rows.append({'String':s,'Learned DFA':'Accept' if pred else 'Reject','Reference':'Accept' if truth else 'Reject','Agree':pred==truth})
                st.dataframe(pd.DataFrame(rows),use_container_width=True)

with tabs[3]:
    if st.button('▶ Train ML Models',type='primary'):
        with st.spinner('Training classifiers and evaluating the DFA...'):
            X=extract_features(strings,alphabet); ix=np.arange(len(labels)); tr,te=train_test_split(ix,test_size=.2,random_state=42,stratify=labels)
            models=train_and_evaluate(X[tr],np.array(labels)[tr],X[te],np.array(labels)[te])
            dfa_pred=np.array([int(current_learned().accepts(strings[i])) if current_learned() else int(ref.accepts(strings[i])) for i in te])
            dm=metrics(np.array(labels)[te],dfa_pred)
        st.session_state.ml_results=models; st.session_state.dfa_metrics=dm; st.session_state.ml_truth=np.array(labels)[te]
        st.session_state.dfa_pred=dfa_pred
    if 'ml_results' in st.session_state:
        models=st.session_state.ml_results; dm=st.session_state.dfa_metrics
        rows=[{'Model':'Learned DFA',**{k:dm[k] for k in ['Accuracy','Precision','Recall','F1-Score']}}]+[{'Model':n,'Accuracy':v['accuracy'],'Precision':v['precision'],'Recall':v['recall'],'F1-Score':v['f1_score']} for n,v in models.items()]
        st.dataframe(pd.DataFrame(rows).style.format({k:'{:.3f}' for k in ['Accuracy','Precision','Recall','F1-Score']}),use_container_width=True)
        chart=pd.DataFrame(rows).melt('Model',var_name='Metric',value_name='Score'); st.plotly_chart(px.bar(chart,x='Metric',y='Score',color='Model',barmode='group',template='plotly_white',range_y=[0,1]),use_container_width=True)
        best=max(models,key=lambda n:models[n]['f1_score']); c1,c2=st.columns(2)
        with c1: st.subheader(f'Best ML: {best}'); st.dataframe(pd.DataFrame(models[best]['confusion_matrix'],index=['True reject','True accept'],columns=['Pred reject','Pred accept']))
        with c2: st.subheader('DFA confusion matrix'); st.dataframe(pd.DataFrame(dm['confusion_matrix'],index=['True reject','True accept'],columns=['Pred reject','Pred accept']))
    else: st.info('Train three scikit-learn models and compare them with the learned DFA.')

with tabs[4]:
    if st.button('▶ Run Experiments',type='primary'):
        with st.spinner('Running size and noise experiments...'):
            rng=random.Random(925); test=random_words(alphabet,maxlen,500,925)
            sizes=[20,50,100,200,500]; size_rows=[]
            for size in sizes:
                xx,yy=generate_samples(language,size//2,size-size//2,maxlen,0.0)
                d=learn_dfa([s for s,y in zip(xx,yy) if y],[s for s,y in zip(xx,yy) if not y],alphabet)
                size_rows.append({'Training size':size,'Accuracy':sum(d.accepts(s)==ref.accepts(s) for s in test)/len(test),'States':d.num_states()})
            noise_rows=[]
            for nf in [0,.05,.1,.15,.2]:
                xx,yy=generate_samples(language,100,100,maxlen,nf)
                d=learn_dfa([s for s,y in zip(xx,yy) if y],[s for s,y in zip(xx,yy) if not y],alphabet)
                dacc=sum(d.accepts(s)==ref.accepts(s) for s in test)/len(test)
                X=extract_features(xx,alphabet); tx=extract_features(test,alphabet); clf=RandomForestClassifier(n_estimators=60,random_state=42).fit(X,yy)
                noise_rows.extend([{'Noise':nf,'Model':'DFA','Accuracy':dacc},{'Noise':nf,'Model':'Random Forest','Accuracy':accuracy_score([int(ref.accepts(s)) for s in test],clf.predict(tx))}])
        st.session_state.size_exp=pd.DataFrame(size_rows); st.session_state.noise_exp=pd.DataFrame(noise_rows)
    if 'size_exp' in st.session_state:
        c1,c2=st.columns(2)
        with c1: st.subheader('Effect of training-set size'); st.plotly_chart(px.line(st.session_state.size_exp,x='Training size',y='Accuracy',markers=True,hover_data=['States'],range_y=[0,1],template='plotly_white'),use_container_width=True)
        with c2: st.subheader('Effect of label noise'); st.plotly_chart(px.line(st.session_state.noise_exp,x='Noise',y='Accuracy',color='Model',markers=True,range_y=[0,1],template='plotly_white'),use_container_width=True)
    else: st.info('Run the experiments to compare sample size and label noise effects.')

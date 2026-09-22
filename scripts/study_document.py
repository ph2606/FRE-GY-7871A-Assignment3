"""Render shared prose and audited tables to LaTeX and a runnable notebook."""
from pathlib import Path
import hashlib,json,re
import nbformat as nbf
import pandas as pd
from study_content import sections,SOCIAL_EXAMPLES,READINGS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'


def esc(value):
    chars={'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
           '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(chars.get(c,c) for c in str(value))


def latex_prose(text):
    """Render the deliberately small Markdown subset used by shared prose."""
    protected=[]
    def keep(s):
        token=f'ZZTOKEN{len(protected)}ZZ';protected.append(s);return token
    text=re.sub(r'\$\$(.*?)\$\$',lambda m:keep(r'\['+m[1]+r'\]'),text,flags=re.S)
    # A digit-leading formula continues directly with a variable/operator;
    # currency amounts are followed by whitespace or punctuation.
    text=re.sub(r'\$(\d+(?:\.\d+)?[A-Za-z_(\\][^$\n]*?)\$',lambda m:keep('$'+m[1]+'$'),text)
    # Other inline math starts with a letter, backslash, or minus; currency stays text.
    text=re.sub(r'(?<!\$)\$([A-Za-z\\-][^$\n]*?)\$',lambda m:keep('$'+m[1]+'$'),text)
    text=re.sub(r'\[([^]]+)\]\((https?://[^)]+)\)',
                lambda m:keep(r'\href{'+esc(m[2])+r'}{'+esc(m[1])+'}'),text)
    text=re.sub(r'`([^`]+)`',lambda m:keep(r'\texttt{'+esc(m[1])+'}'),text)
    text=re.sub(r'\*\*(.*?)\*\*',lambda m:keep(r'\textbf{'+esc(m[1])+'}'),text)
    text=re.sub(r'\*([^*]+)\*',lambda m:keep(r'\emph{'+esc(m[1])+'}'),text)
    text=esc(text)
    for i,value in enumerate(protected):text=text.replace(f'ZZTOKEN{i}ZZ',value)
    return text


def number(v,digits=3):
    return '—' if pd.isna(v) else f'{v:,.{digits}f}'


def table(headers,rows,layout=None,size='footnotesize',raw=False,long=False):
    layout=layout or 'l'+'r'*(len(headers)-1)
    env='longtable' if long else 'tabular'
    lines=[r'\par\smallskip\noindent','{\\'+size,r'\begin{'+env+'}{'+layout+'}',r'\toprule',
           ' & '.join(esc(x) for x in headers)+r' \\',r'\midrule']
    if long:lines.append(r'\endhead')
    for row in rows:
        lines.append(' & '.join(str(x) if raw else esc(x) for x in row)+r' \\[3pt]')
    lines += [r'\bottomrule',r'\end{'+env+'}',r'}\par']
    return '\n'.join(lines)


def fig(name,width='.96'):
    return r'\begin{center}\includegraphics[width='+width+r'\linewidth]{outputs/'+name+r'.pdf}\end{center}'


def displays():
    t1=pd.read_csv(OUT/'paper_table1.csv').fillna('')
    t2=pd.read_csv(OUT/'paper_table2.csv')
    t3=pd.read_csv(OUT/'paper_table3.csv')
    match=pd.read_csv(OUT/'paper_matching.csv').set_index('high_date')
    result={}
    result['data']=table(['Market','Original paper','Public-data implementation'],[
       ['Treasuries','Fitted off-the-run par yields','GSW fitted par; CMT sensitivity'],
       ['Oil','12-month futures, dollars','Nearby WTI dollars; spot/nearby percent'],
       ['Gold','Gold dollar price','GLD percentage return'],
       ['Credit','BBB/high-yield spreads','ICE BofA option-adjusted spreads'],
       ['Liquidity','On-the-run premium','Unavailable; not substituted']
    ],layout=r'p{.12\linewidth}p{.34\linewidth}p{.43\linewidth}')
    result['trends']=fig('study_oil','.88')+r'\newpage'+fig('study_markets','.94')
    rows=[]
    for e in t1.itertuples():
        day=e.market_date[:10]
        low=match.at[day,'low_date'] if day in match.index else ''
        comparison=str(low)[5:10] if pd.notna(low) and low else 'Unavailable'
        information=r'\href{'+esc(e.source_url)+'}{'+esc(e.description)+'}. '+esc(e.event_date[5:]+' '+e.weekday[:3]+', '+e.cash_window)+'.'
        rows.append([esc(day[5:]),str(int(e.war_articles)),number(e.intensity,2),esc(comparison),information])
    result['table1']=fig('study_news','.88')+r'\textbf{Table 1. NLP high-news sessions and illustrative source headlines (2026).}'+table(
        ['H session','Articles','Per day','L session','Illustrative headline, source date and clock'],rows,
        layout=r'p{.08\linewidth}rrp{.09\linewidth}p{.56\linewidth}',raw=True,long=True)
    rows=[]
    for x in t2.itertuples():
        effects=[]
        for name in ['omega1','omega2','pooled']:
            effect=getattr(x,name+'_effect');t=getattr(x,name+'_abs_t_hc1')
            effects.append('—' if pd.isna(effect) else number(effect)+r' ('+number(t,2)+')')
        flag=('W' if x.weak_reference_variance_shift else '') if x.available else '—'
        rows.append([esc(x.label),esc(x.unit),str(int(x.n_high)) if x.available else '—',*effects,flag])
    result['table2']=r'\textbf{Table 2. Response to a conditional 25-bp two-year-yield decline.}'+table(
        ['Outcome','Unit','Pairs','Reference IV','Outcome IV','Pooled IV',''],rows,
        layout=r'p{.245\linewidth}lrrrrl',raw=True,size='footnotesize')+r'\par\footnotesize Effects with absolute HC1 t-statistics in parentheses. W: uncertain positive reference-variance shift. All available rows are conditional diagnostics; no strong-identification significance stars are used.\normalsize'
    rows=[]
    for x in t3.itertuples():
        label=x.label
        unit=x.unit_squared.replace('^2','²')
        rows.append([label,unit,number(x.variance_low,5),number(x.variance_high,5),number(x.predicted_variance_change,5),
            number(x.high_variance_share_pct,2),number(x.all_variance_share_pct,2)])
    result['table3']=r'\textbf{Table 3. Mean squares and conditional variance shares.}'+table(
        ['Outcome','Unit²','L','H','Predicted','H share %','All share %'],rows,
        layout=r'p{.24\linewidth}lrrrrr',size='footnotesize')+r'\par\footnotesize Squared units apply to L, H and predicted variance; both share columns are percentages. No causal lower bound is established.\normalsize'
    robust=pd.read_csv(OUT/'paper_robustness.csv');rows=[]
    base=t2.set_index('variable')
    rows.append(['Main fitted curve',str(int(base.at['ten_year','n_high']))]+[number(base.at[k,'pooled_effect'],2) for k in ['ten_year','sp500','wti_spot','brent_spot']])
    for spec,a in robust.groupby('specification',sort=False):
        a=a.set_index('variable')
        rows.append([spec,str(int(a.at['ten_year','n_high']))]+[number(a.at[k,'pooled_effect'],2) for k in ['ten_year','sp500','wti_spot','brent_spot']])
    result['robust']=fig('study_identification')+table(['Specification','2y/10y pairs','10y pp','S&P %','WTI %','Brent %'],rows,
            layout=r'p{.29\linewidth}rrrrr')+r'\par\footnotesize Pair counts shown are for the ten-year outcome; oil can have fewer complete pairs.\normalsize'
    benchmark=pd.read_csv(OUT/'benchmark_expectations.csv');rows=[]
    for x in benchmark.itertuples():
        expected='Ambiguous' if pd.isna(x.iran_expected_sign) else ('Up' if x.iran_expected_sign>0 else 'Down')
        historical=number(x.original_2003_effect)
        if x.variable=='gold_gld':historical+=' $/oz'
        rows.append([x.label,x.unit_2026,historical,expected,number(x.iran_plus25bp_scenario)])
    result['benchmark']=table(['Outcome','Iran unit','Iraq −25 bp','Iran hypothesis','Iran +25 bp'],rows,
        layout=r'p{.29\linewidth}lrrr')+r'\par\footnotesize Historical oil is a 12-month contract in dollars per barrel; Iran nearby oil uses a different maturity. Historical gold is dollars per ounce; Iran is GLD percent. Scenarios rescale the same estimates; they do not identify shock direction.\normalsize'
    directional=pd.read_csv(OUT/'directional_regime_markets.csv');rows=[]
    for regime,frame in directional.groupby('regime',sort=False):
        a=frame.set_index('outcome')
        cells=[number(a.at[k,'mean_change'],3)+' ('+str(int(a.at[k,'observations']))+')' for k in ['ten_year','wti_spot','brent_spot','sp500']]
        rows.append([regime,str(int(frame.sessions.iloc[0])),*cells])
    result['direction']=table(['Text state','Sessions','10y pp','WTI %','Brent %','S&P %'],rows,
        layout=r'p{.27\linewidth}rrrrr')+r'\par\footnotesize Conditional daily means; available observations in parentheses. Labels describe headline language. These are not identified causal responses.\normalsize'
    phrases=json.loads((OUT/'phrase_audit_summary.json').read_text())
    rows=[[key.replace('_',' ').replace(' cue',''),str(value)] for key,value in phrases['cue_document_counts'].items()]
    result['phrases']=table(['Review cue','Documents'],rows,layout=r'p{.74\linewidth}r')+r'\par\footnotesize Cues overlap and flag interpretation risks. They are not validated labels or war-ending probabilities.\normalsize'
    news=json.loads((OUT/'duration_manifest.json').read_text())
    rows=[[k,f'{v:,}'] for k,v in news['guardian_categories'].items()]
    rows += [['GDELT: '+k,f'{v:,}'] for k,v in news['gdelt_categories'].items()]
    result['news']=table(['Classification basis and category','Items / URLs'],rows,layout=r'p{.76\linewidth}r')+fig('study_news','.88')
    r=pd.read_csv(OUT/'event_reaction_windows.csv')
    a=r.loc[r.friday_sunday & r.outcome.eq('sp500')]
    rows=[[x.event_date[5:],x.reaction_session[5:],x.following_session[5:],number(x.reaction_change,2),number(x.following_change,2)] for x in a.itertuples()]
    result['timing']=fig('study_timing')+r'\textbf{Friday–Sunday chronology: S\&P 500 daily returns (\%).}'+table(
        ['Event date','Reaction session','Following session','Reaction %','Following %'],rows)
    lp=pd.read_csv(OUT/'duration_local_projections.csv')
    a=lp.loc[lp.horizon.eq(-1) & lp.q_value.lt(.05)]
    rows=[[x.outcome.replace('_',' '),x.feature,number(x.coefficient,3),x.unit,number(x.q_value,3),str(x.n)] for x in a.itertuples()]
    result['alternative']=fig('study_alternative')+r'\textbf{Prior-return timing checks with adjusted q below 0.05.}'+table(
        ['Market','News category','Prior response','Unit','q','N'],rows)+r'\par\footnotesize These dependent returns precede the archive-availability session. They are not forward trading returns.\normalsize'
    rows=[[esc(date),r'\href{https://truthsocial.com/@realDonaldTrump/'+post+'}{Original post.} '+esc(reading)] for date,post,reading in SOCIAL_EXAMPLES]
    result['social']=table(['Date (ET)','Target, condition and interpretation'],rows,layout=r'p{.13\linewidth}p{.80\linewidth}',raw=True,long=True)
    rows=[[who+'. '+title,lesson] for who,title,lesson in READINGS]
    result['readings']=table(['Supplied reference','Application to this corpus'],rows,layout=r'p{.42\linewidth}p{.51\linewidth}',long=True)
    return result


def build_notebook(blocks):
    notebook=nbf.v4.new_notebook()
    notebook.cells=[nbf.v4.new_markdown_cell('# Iran war risk and global financial markets in 2026\n\n'
        '**Panagiotis Housos · ph2606**  \nFRE-GY 7871A · NLP and the Investment Process · Assignment 3  \n'
        'Main window: **28 February–18 September 2026**. Prepared 21 September 2026.  \n'
        'This notebook contains saved outputs. Follow the README to acquire the local inputs before executing it.')]
    for i,b in enumerate(blocks,1):
        notebook.cells.append(nbf.v4.new_markdown_cell(f'## {i}. '+b['title']+'\n\n'+b['text']))
        if b['code']:notebook.cells.append(nbf.v4.new_code_cell(b['code']))
    notebook.metadata={'kernelspec':{'display_name':'Python 3 (Assignment 3)','language':'python','name':'assignment3'},
        'language_info':{'name':'python','version':'3.12.10'}}
    nbf.validate(notebook)
    nbf.write(notebook,ROOT/'assignment3.ipynb')


def build(write_notebook=True):
    diagnostics=json.loads((OUT/'study_diagnostics.json').read_text())
    for path,digest in diagnostics['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    blocks=sections(ROOT);extra=displays()
    parts=[]
    for i,b in enumerate(blocks,1):
        if b['page']:parts.append(r'\clearpage')
        parts.append(r'\section*{'+str(i)+'. '+esc(b['title'])+'}')
        parts.append(latex_prose(b['text']))
        if b['display']:parts.append(extra[b['display']])
    template=(ROOT/'scripts/report_template.tex').read_text(encoding='utf-8')
    assert template.count('%%CONTENT%%')==1
    (ROOT/'report.tex').write_text(template.replace('%%CONTENT%%','\n\n'.join(parts)),encoding='utf-8')
    if write_notebook:build_notebook(blocks)
    print('Built report source'+(' and executable notebook' if write_notebook else '')+' from verified analysis outputs.')


if __name__=='__main__':build()

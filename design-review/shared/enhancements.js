/* Shared interactions for the three editions preserving the original project page. */
const paper = window.PAPER_DATA;
const legacy = window.LEGACY_DATA;
const normalizeModel = name => name.toLowerCase().replace(/[^a-z0-9]/g, '');
const legacyByName = new Map(legacy.leaderboard.models.map(m => [normalizeModel(m.model), m]));
const percentTree = value => {
  if (typeof value === 'number') return value * 100;
  if (!value || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, percentTree(v)]));
};
const matchedLegacy = new Set();
lbModels = paper.models.map(m => {
  const old = legacyByName.get(normalizeModel(m.model));
  if (old) matchedLegacy.add(old.model);
  return {model:m.model, type:m.type,
    real:{...m, dim:m, domain:old ? percentTree(old.real.domain) : {}},
    synthetic:old ? percentTree(old.synthetic) : null};
});
legacy.leaderboard.models.filter(m => !matchedLegacy.has(m.model)).forEach(m => {
  lbModels.push({model:m.model, type:m.type, real:null, synthetic:percentTree(m.synthetic)});
});
REAL_COLS.forEach(c => {
  c.label = c.label.replace('tok', 'word');
  c.group = c.group.replace('tok', 'word').replace('Domain F1', 'Domain F1 · public v1');
});
REAL_COLS.splice(3, 0,
  {key:'fpr', label:'FPR ↓', group:'Overall', src:'overall', lower:true},
  {key:'zor', label:'ZOR ↓', group:'Overall', src:'overall', lower:true});
SYNTH_COLS.forEach(c => c.group = c.group.replace('Overall', 'Public v1 · token metrics'));

getVal = function(model, col) {
  const source = currentTab === 'real' ? model.real : model.synthetic;
  if (!source) return null;
  return col.src === 'overall' ? source[col.key] ?? null : source[col.src]?.[col.key] ?? null;
};
getBestValues = function(rows, cols) {
  return Object.fromEntries(cols.map((col, i) => {
    const values = rows.map(m => getVal(m,col)).filter(v => v != null);
    return [i, col.lower ? Math.min(...values) : Math.max(...values)];
  }));
};
render = function() {
  const cols = currentTab === 'real' ? REAL_COLS : SYNTH_COLS;
  const source = currentTab === 'real' ? 'real' : 'synthetic';
  const rows = lbModels.filter(m => m[source] && (currentFilter === 'all' || m.type === currentFilter));
  const sortCol = cols[sortColIdx ?? 2];
  rows.sort((a,b) => {
    const av=getVal(a,sortCol), bv=getVal(b,sortCol);
    if (av==null) return bv==null ? 0 : 1;
    if (bv==null) return -1;
    return (sortColIdx===null || sortDir==='desc') ? bv-av : av-bv;
  });
  const best = getBestValues(rows,cols);
  const groups=[];
  cols.forEach(c => {
    if (groups.at(-1)?.name===c.group) groups.at(-1).count++;
    else groups.push({name:c.group,count:1});
  });
  document.getElementById('lb-thead').innerHTML = '<tr><th class="sticky-1" rowspan="2">Rank</th><th class="sticky-2" rowspan="2">Model</th><th class="sticky-3" rowspan="2">Type</th>'+groups.map(g=>`<th colspan="${g.count}" class="col-sep">${g.name}</th>`).join('')+'</tr><tr>'+cols.map((c,i)=>`<th scope="col" tabindex="0" role="button" aria-label="Sort by ${c.key}" aria-sort="${sortColIdx===i?(sortDir==='asc'?'ascending':'descending'):'none'}" class="${c.sep?'col-sep ':''}${sortColIdx===i?'sort-'+sortDir:''}" onclick="sortBy(${i})" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();sortBy(${i})}">${c.label}</th>`).join('')+'</tr>';
  document.getElementById('lb-tbody').innerHTML = rows.map((m,i)=>'<tr>'+`<td class="sticky-1 rank-medal">${String(i+1).padStart(2,'0')}</td><td class="sticky-2 align-left">${escHtml(m.model)}</td><td class="sticky-3"><span class="badge badge-${m.type}">${m.type==='open'?'Open':'Closed'}</span></td>`+cols.map((c,j)=>{
    const v=getVal(m,c); return `<td class="${c.sep?'col-sep-body ':''}${v!=null && v===best[j]?'best':''}">${fmt(v)}</td>`;
  }).join('')+'</tr>').join('');
  const note=document.getElementById('lb-protocol-note');
  if(note) note.textContent=currentTab==='real' ? 'Real scores: current paper, Word-level percentages. FPR and ZOR: lower is better. Domain columns retain the public v1 evaluation; they are not directly comparable with current Word scores.' : 'Synthetic scores: retained public v1 release, token-level percentages. These are not the new paper’s real-test Word metrics. Model coverage differs between releases.';
};
sortBy = function(i) {
  const cols=currentTab==='real'?REAL_COLS:SYNTH_COLS;
  sortDir=sortColIdx===i?(sortDir==='desc'?'asc':'desc'):(cols[i].lower?'asc':'desc');
  sortColIdx=i; render();
};
render();

downloadTable = async function() {
  const button=document.querySelector('.download-btn');
  const oldLabel=button.innerHTML;
  button.disabled=true; button.textContent='Preparing image…';
  const exportRoot=document.createElement('div');
  const source=document.getElementById('lb-table');
  const clone=source.cloneNode(true);
  exportRoot.style.cssText='position:fixed;top:0;left:-15000px;padding:30px;background:#ffffff;color:#152038;z-index:-1';
  exportRoot.style.width=(source.scrollWidth+60)+'px';
  exportRoot.innerHTML='<h3 style="font:600 20px Manrope;margin:0 0 10px">DetailVerifyBench</h3><p style="font:12px Manrope;margin:0 0 20px">'+(currentTab==='real'?'Real test · current manuscript · Word-level percentages. Domain columns: public v1.':'Synthetic · public v1 · token-level percentages.')+'</p>';
  exportRoot.append(clone); document.body.append(exportRoot);
  clone.querySelectorAll('th,td').forEach(c=>{c.style.position='static';c.style.color='#253041';c.style.background=c.tagName==='TH'?'#edf1f7':'#fff';});
  try {
    await document.fonts.ready;
    const canvas=await html2canvas(exportRoot,{scale:2,backgroundColor:'#fff',windowWidth:exportRoot.scrollWidth+100,logging:false});
    const link=document.createElement('a');
    link.download=`DetailVerifyBench_${currentTab}_${currentFilter}.png`;
    link.href=canvas.toDataURL('image/png'); link.click();
    button.textContent='Image saved';
  } catch(error) {
    button.textContent='Export unavailable — try a local server';
    console.error('Leaderboard image export failed:',error);
  } finally {
    exportRoot.remove(); button.disabled=false;
    setTimeout(()=>button.innerHTML=oldLabel,2500);
  }
};

const evidence=[
  ['The cuffs are white.','The image contradicts the color stated in the caption.'],
  ['No bayonet is attached.','The rifle is visible, but the claimed attachment is absent.'],
  ['The soldier is kneeling or crouching.','The caption incorrectly describes the visible body posture.'],
  ['The soldier looks forward or downward.','The direction of the gaze contradicts the claimed backward glance.'],
  ['The soldier stands clear of the balustrade.','The caption invents physical contact with the stone railing.']
];
const locator=document.createElement('div');locator.className='evidence-locator';locator.setAttribute('aria-hidden','true');
locator.innerHTML=evidence.map((_,i)=>`<span class="${i===0?'active':''}">${i+1}</span>`).join('');
document.querySelector('.evidence-image').append(locator);
const crop=document.createElement('div');crop.className='selected-evidence';
crop.innerHTML='<img src="../shared/cases/teaser-evidence-1.webp" alt="White cuffs: detail from the paper’s figure"><div><span>SELECTED VISUAL EVIDENCE</span><p>The cuffs are white.</p><small>Inspection crop from Figure 1</small></div>';
document.querySelector('.evidence-image').append(crop);
document.querySelectorAll('[data-evidence]').forEach(button=>button.addEventListener('click',()=>{
  const index=Number(button.dataset.evidence);
  document.querySelectorAll('[data-evidence]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  locator.querySelectorAll('span').forEach((s,i)=>s.classList.toggle('active',index===i));
  document.getElementById('evidence-count').textContent=`0${index+1} / 05`;
  document.getElementById('evidence-fact').textContent=evidence[index][0];
  document.getElementById('evidence-note').textContent=evidence[index][1];
  crop.querySelector('img').src=`../shared/cases/teaser-evidence-${index+1}.webp`;
  crop.querySelector('img').alt=evidence[index][0];
  crop.querySelector('p').textContent=evidence[index][0];
}));
document.getElementById('hide-evidence').addEventListener('click',function(){
  const hidden=document.querySelector('.evidence-caption').classList.toggle('annotations-hidden');
  this.textContent=hidden?'Reveal annotations':'Hide annotations';
  this.setAttribute('aria-pressed',String(hidden));
});

const reasoning=[
  '“September is 14, and October is 9.”',
  '14 − 5 = 9. The reasoning now follows the caption’s arithmetic instead of the observed chart.',
  'Final model output: NO HALLUCINATION. The incorrect month remains unmarked.'
];
document.querySelectorAll('[data-step]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('[data-step]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  document.getElementById('reason-output').textContent=reasoning[Number(button.dataset.step)];
}));
document.querySelectorAll('[data-reward]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('[data-reward]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  document.getElementById('reward-number').textContent=Number(button.dataset.reward).toFixed(2);
}));
const pipelineCaptions=[
  '1,000 detail-rich images across five domains → Gemini-3-Pro candidate captions.',
  'Human correction → text-diff boundary extraction → cross-checking of word-level spans.',
  'The complete construction pipeline: 623 captions with hallucinations and 377 without.'
];
setPipeStep=function(i){
  document.getElementById('pipeline-view').dataset.stage=String(i);
  document.getElementById('pipeline-caption').textContent=pipelineCaptions[i];
  document.querySelectorAll('#pipeline .pipe-tab').forEach((b,j)=>{b.classList.toggle('active',i===j);b.setAttribute('aria-pressed',String(i===j));});
};

const dialog=document.getElementById('figure-dialog');
const dialogImage=document.getElementById('dialog-figure');
const pdfLink=document.createElement('a');pdfLink.className='figure-pdf';pdfLink.textContent='Open original figure PDF ↗';pdfLink.target='_blank';pdfLink.rel='noopener';dialog.insertBefore(pdfLink,dialog.querySelector('.figure-scroll'));
function showFigure(path,alt='Enlarged source image'){
  dialogImage.src=path;dialogImage.alt=alt;dialogImage.classList.remove('zoomed');
  pdfLink.hidden=!path.includes('/paper/');
  if(!pdfLink.hidden)pdfLink.href=path.replace(/\.webp$/,'.pdf');else pdfLink.removeAttribute('href');
  dialog.showModal();
}
document.addEventListener('click',event=>{
  const figure=event.target.closest('[data-figure]');
  if(figure)showFigure(figure.dataset.figure,figure.querySelector('img')?.alt||figure.textContent.trim());
  else if(event.target.matches('.ex-viewer-img img,.det-panel-img img,.inj-panel-img img,#pipe-img'))showFigure(event.target.src,event.target.alt);
});
document.getElementById('close-figure').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
dialogImage.addEventListener('click',()=>dialogImage.classList.toggle('zoomed'));
dialogImage.title='Click to inspect at a larger scale';

document.getElementById('copy-citation').addEventListener('click',async function(){
  const value=document.getElementById('bibtex').textContent;
  try {
    if(navigator.clipboard && window.isSecureContext)await navigator.clipboard.writeText(value);
    else {
      const area=document.createElement('textarea');area.value=value;area.style.cssText='position:fixed;left:-10000px';document.body.append(area);area.select();
      if(!document.execCommand('copy'))throw new Error('Clipboard unavailable');area.remove();
    }
    this.textContent='Copied';
  }catch(error){this.textContent='Select BibTeX to copy';const range=document.createRange();range.selectNodeContents(document.getElementById('bibtex'));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);}
  setTimeout(()=>this.textContent='Copy BibTeX',2500);
});
document.getElementById('coverage-table').innerHTML='<div class="table-wrapper"><table><thead><tr><th>Model</th><th>Valid outputs / 1,000</th><th>Coverage</th></tr></thead><tbody>'+paper.models.map((m,i)=>`<tr><td>${escHtml(m.model)}</td><td>${paper.validOutputs[i]}</td><td>${(paper.validOutputs[i]/10).toFixed(1)}%</td></tr>`).join('')+'</tbody></table></div>';

document.querySelectorAll('.det-model-option').forEach(option=>{
  option.tabIndex=0;option.setAttribute('role','checkbox');option.setAttribute('aria-checked',String(option.classList.contains('selected')));
  option.addEventListener('click',()=>option.setAttribute('aria-checked',String(option.classList.contains('selected'))));
  option.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();option.click();}});
});
if('IntersectionObserver' in window){
  const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){document.querySelectorAll('.nav-links a').forEach(a=>a.classList.toggle('active',a.hash==='#'+entry.target.id));}}),{rootMargin:'-10% 0px -65% 0px'});
  document.querySelectorAll('main section').forEach(s=>observer.observe(s));
}

/* Preserved project-page interactions for editions 01 and 03. */
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
  if(c.group==='Overall')c.group='Word Metrics (%)';
  c.group = c.group.replace('tok', 'word').replace('Domain F1', 'Domain Token F1');
});
REAL_COLS.splice(3, 0,
  {key:'fpr', label:'FPR ↓', group:'Diagnostics (%)', src:'overall', lower:true},
  {key:'zor', label:'ZOR ↓', group:'Diagnostics (%)', src:'overall', lower:true});
SYNTH_COLS.forEach(c => c.group = c.group.replace('Overall', 'Token Metrics (%)'));

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
  exportRoot.innerHTML='<h3 style="font:600 20px Manrope;margin:0 0 10px">DetailVerifyBench</h3><p style="font:12px Manrope;margin:0 0 20px">'+(currentTab==='real'?'Real test · Word-level metrics (%). Domain columns: token F1 (%).':'Synthetic · Token-level metrics (%).')+'</p>';
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

document.querySelectorAll('[data-reward]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('[data-reward]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  document.getElementById('reward-number').textContent=Number(button.dataset.reward).toFixed(2);
}));
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

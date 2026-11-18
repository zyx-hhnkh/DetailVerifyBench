/* Figure 2's five operations, presented as a sequential card deck. */
(()=>{
  const domains=[['GUI','gui.png'],['Nature','nature.jpg'],['Chart','chart.png'],['Movie','movie.jpg'],['Poster','poster.jpg']];
  const steps=[
    {title:'Image Selection',stage:'Dataset Initialization',body:`<div class="construction-gallery">${domains.map(([name,file])=>`<figure><img src="assets/shared/construction/${file}" alt="${name} benchmark image" loading="lazy"><figcaption>${name}</figcaption></figure>`).join('')}</div><div class="construction-count"><strong>1,000</strong><span>images · 200 per domain</span></div>`},
    {title:'Caption Generation',stage:'Dataset Initialization',body:'<div class="construction-generator"><span>Domain-Specific Prompts</span><span aria-hidden="true">→</span><strong>Gemini-3-Pro</strong></div><div class="construction-draft"><span>Draft Caption</span><p>… with <em>red</em> cuffs …</p></div>'},
    {title:'Human Correction',stage:'Hallucination Annotation',body:'<div class="construction-correction"><div><span>Original Caption</span><p>… with <del>red</del> cuffs …</p></div><span class="construction-arrow" aria-hidden="true">→</span><div><span>Corrected Caption</span><p>… with <ins>black</ins> cuffs …</p></div></div>'},
    {title:'Text Diff',stage:'Hallucination Annotation',body:'<div class="construction-diff"><div><span>Original</span><p>… with <mark>red</mark> cuffs …</p></div><div><span>Corrected</span><p>… with <b>black</b> cuffs …</p></div></div><div class="construction-span"><span>Reference Span</span><code>&lt;HALLUCINATION&gt;<b>red</b>&lt;/HALLUCINATION&gt;</code></div>'},
    {title:'Cross-Check',stage:'Hallucination Annotation',body:'<div class="construction-output"><span class="construction-check" aria-hidden="true">✓</span><div><strong>1,818</strong><span>word-level error spans</span></div></div><div class="construction-split"><div><b>623</b><span>captions with errors</span></div><div><b>377</b><span>error-free captions</span></div></div>'}
  ];
  document.querySelectorAll('[data-construction]').forEach(root=>{
    root.classList.add('construction-flow');
    root.setAttribute('role','region');
    root.setAttribute('aria-label','Benchmark construction: five steps');
    root.innerHTML=`<div class="construction-deck">${steps.map((step,i)=>`<article class="construction-card" data-build-node="${i}" aria-hidden="${i!==0}"><header class="construction-heading"><div><span class="construction-stage">${step.stage}</span><h3>${step.title}</h3></div><span class="construction-number">0${i+1}</span></header><div class="construction-content">${step.body}</div><div class="construction-card-foot"><span>${['Selected Images','Caption Drafts','Corrected Captions','Reference Spans','Annotated Benchmark'][i]}</span><span>0${i+1} / 05</span></div></article>`).join('')}</div>
      <div class="construction-steps" role="group" aria-label="Construction steps">${steps.map((step,i)=>`<button class="construction-step" data-build-step="${i}" aria-pressed="${i===0}"><span>0${i+1}</span>${step.title}</button>`).join('')}</div>
      <div class="construction-controls"><button class="construction-prev" type="button" aria-label="Previous construction step">←</button><div class="construction-progress" aria-hidden="true"><i></i></div><button class="construction-next" type="button" aria-label="Next construction step">→</button><button class="construction-play" type="button">Play</button></div>`;
    const cards=[...root.querySelectorAll('[data-build-node]')];
    const controls=[...root.querySelectorAll('[data-build-step]')];
    const play=root.querySelector('.construction-play');
    const prev=root.querySelector('.construction-prev');
    const next=root.querySelector('.construction-next');
    const reduced=window.matchMedia?.('(prefers-reduced-motion: reduce)').matches||false;
    let current=0,timer=null,transitionTimer=null,started=false;
    function select(index,animate=true){
      const old=current,changed=index!==old;
      current=index;clearTimeout(transitionTimer);
      root.dataset.buildStep=String(index);
      root.dataset.buildDirection=index>old?'forward':'backward';
      root.style.setProperty('--construction-progress',`${(index+1)/cards.length*100}%`);
      cards.forEach((card,i)=>{
        card.classList.toggle('is-current',i===index);
        card.classList.toggle('is-prev',i===index-1);
        card.classList.toggle('is-next',i===index+1);
        card.classList.toggle('is-entering',animate&&changed&&!reduced&&i===index);
        card.classList.toggle('is-leaving',animate&&changed&&!reduced&&i===old);
        card.setAttribute('aria-hidden',String(i!==index));card.inert=i!==index;
      });
      controls.forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));
      prev.disabled=index===0;next.disabled=index===cards.length-1;
      transitionTimer=setTimeout(()=>cards.forEach(card=>card.classList.remove('is-entering','is-leaving')),700);
    }
    function stop(){
      clearTimeout(timer);timer=null;root.classList.remove('is-playing');
      play.textContent=current===cards.length-1?'Replay':'Play';
    }
    function advance(){
      timer=setTimeout(()=>{
        if(current===cards.length-1){stop();return;}
        select(current+1);advance();
      },3000);
    }
    function start(){
      stop();if(current===cards.length-1)select(0);
      root.classList.add('is-playing');play.textContent='Pause';advance();
    }
    function choose(index){started=true;stop();select(index);play.textContent=index===cards.length-1?'Replay':'Play';}
    controls.forEach((button,i)=>button.addEventListener('click',()=>choose(i)));
    prev.addEventListener('click',()=>choose(Math.max(0,current-1)));
    next.addEventListener('click',()=>choose(Math.min(cards.length-1,current+1)));
    play.addEventListener('click',()=>{started=true;timer?stop():start();});
    select(0,false);
    if('IntersectionObserver' in window){
      const observer=new IntersectionObserver(entries=>{
        entries.forEach(entry=>{
          if(entry.isIntersecting&&!started){started=true;if(!reduced)start();}
          else if(!entry.isIntersecting)stop();
        });
      },{threshold:.4});
      observer.observe(root);
    }
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  });
})();

/* Shared movie example. Text nodes preserve the selected caption excerpt exactly. */
(()=>{
  const data=window.MOVIE_CASE;
  if(!data)return;
  document.querySelectorAll('[data-movie-case]').forEach(root=>{
    let selected=0,highlighted=true;
    root.innerHTML=`<div class="movie-stage"><div class="movie-visual"><div class="movie-image"><button class="movie-zoom" aria-label="Enlarge movie image"><img src="../shared/cases/movie-0.webp" alt="Movie frame and five enlarged evidence regions"></button>${data.evidence.map((e,i)=>`<button class="movie-hotspot hotspot-${i+1}" data-movie-evidence="${i}" aria-label="Inspect region ${i+1}: ${e.phrase}" aria-pressed="${i===0}"></button>`).join('')}</div><div class="movie-selectors" role="group" aria-label="Visual evidence">${data.evidence.map((e,i)=>`<button data-movie-evidence="${i}" aria-label="Evidence ${i+1}" aria-pressed="${i===0}">${i+1}</button>`).join('')}</div><div class="movie-selected" aria-live="polite"><img src="../shared/cases/teaser-evidence-1.webp" alt="White cuffs"><p>${data.evidence[0].truth}</p></div></div><div class="movie-reading"><div class="movie-toolbar"><h3>Image Caption</h3><button class="movie-toggle" aria-pressed="false">Unhighlight</button></div><div class="movie-caption" tabindex="0" aria-label="Image caption excerpt"></div></div></div><dialog class="movie-dialog"><button class="movie-close" aria-label="Close movie image">Close ×</button><img src="../shared/cases/movie-0.webp" alt="Full movie frame with five evidence crops"><a href="../shared/paper/teaser.pdf" target="_blank" rel="noopener">Figure 1 ↗</a></dialog>`;
    const caption=root.querySelector('.movie-caption');
    const dialog=root.querySelector('.movie-dialog');
    const crop=root.querySelector('.movie-selected img');
    const truth=root.querySelector('.movie-selected p');
    function renderCaption(){
      caption.replaceChildren();
      if(!highlighted){caption.textContent=data.caption;return;}
      data.parts.forEach(part=>{
        if(!part.error){caption.append(document.createTextNode(part.text));return;}
        const mark=document.createElement('mark');mark.textContent=part.text;mark.className='movie-error';
        if(part.evidence!==null){
          mark.dataset.movieSpan=String(part.evidence);mark.tabIndex=0;mark.setAttribute('role','button');
          mark.setAttribute('aria-label',`Inspect evidence: ${part.text}`);
          mark.classList.toggle('selected',part.evidence===selected);
          mark.setAttribute('aria-pressed',String(part.evidence===selected));
          mark.addEventListener('click',()=>select(part.evidence,false));
          mark.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select(part.evidence,false);}});
        }
        caption.append(mark);
      });
    }
    function select(index,followCaption){
      selected=index;
      root.querySelectorAll('[data-movie-evidence]').forEach(button=>{
        const active=Number(button.dataset.movieEvidence)===index;
        button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));
      });
      root.querySelectorAll('[data-movie-span]').forEach(mark=>{
        const active=Number(mark.dataset.movieSpan)===index;
        mark.classList.toggle('selected',active);mark.setAttribute('aria-pressed',String(active));
      });
      crop.src=`../shared/cases/teaser-evidence-${index+1}.webp`;crop.alt=data.evidence[index].truth;
      truth.textContent=data.evidence[index].truth;
      if(followCaption&&highlighted){
        const mark=caption.querySelector(`[data-movie-span="${index}"]`);
        const box=caption.getBoundingClientRect(),target=mark.getBoundingClientRect();
        if(target.top<box.top+12||target.bottom>box.bottom-12)caption.scrollTop+=target.top-box.top-caption.clientHeight/3;
      }
    }
    root.querySelectorAll('[data-movie-evidence]').forEach(button=>button.addEventListener('click',()=>select(Number(button.dataset.movieEvidence),true)));
    root.querySelector('.movie-toggle').addEventListener('click',function(){
      const scroll=caption.scrollTop;highlighted=!highlighted;renderCaption();caption.scrollTop=scroll;
      this.textContent=highlighted?'Unhighlight':'Highlight';this.setAttribute('aria-pressed',String(!highlighted));
    });
    root.querySelector('.movie-zoom').addEventListener('click',()=>dialog.showModal());
    root.querySelector('.movie-close').addEventListener('click',()=>dialog.close());
    dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
    renderCaption();select(0,false);
  });
})();

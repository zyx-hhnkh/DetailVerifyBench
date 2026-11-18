/* Reveal navigation when the reader returns upward. */
(()=>{
  const nav=document.querySelector('body > .site-nav, body > header');
  if(!nav)return;
  nav.classList.add('reading-navigation');
  const links=[...nav.querySelectorAll('a[href^="#"]')].filter(link=>document.getElementById(link.hash.slice(1))&&link.hash!=='#top');
  function activate(hash){
    links.forEach(link=>{
      const active=link.hash===hash;
      link.classList.toggle('active',active);
      if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');
    });
  }
  function updateActive(){
    const readingLine=nav.offsetHeight+60;
    let active='',nearest=-Infinity;
    links.forEach(link=>{
      const section=document.getElementById(link.hash.slice(1));
      const region=section.closest('section')||section;
      if(region.getBoundingClientRect().bottom<=readingLine)return;
      const heading=section.querySelector('h2,h3')||section;
      const top=heading.getBoundingClientRect().top;
      if(top<=readingLine&&top>nearest){active=link.hash;nearest=top;}
    });
    activate(active);
  }
  let navigationTarget=null,navigationTimer,keepNavigationVisible=false;
  let previous=window.scrollY,travel=0,direction=0,scheduled=false;
  function sectionPosition(target){
    if(target===document.body)return 0;
    const bounds=target.getBoundingClientRect();
    const gap=window.innerWidth<768?16:24;
    const top=window.scrollY+bounds.top-nav.offsetHeight-gap;
    return Math.max(0,Math.min(top,document.documentElement.scrollHeight-window.innerHeight));
  }
  function cancelNavigation(){
    navigationTarget=null;
    clearTimeout(navigationTimer);
  }
  function finishNavigation(){
    if(!navigationTarget)return;
    const top=sectionPosition(navigationTarget);
    cancelNavigation();
    // Images loaded during the smooth scroll can change the heading's position.
    if(Math.abs(window.scrollY-top)>1)window.scrollTo({top,behavior:'instant'});
    previous=top;travel=0;direction=0;
  }
  window.addEventListener('scrollend',finishNavigation);
  ['wheel','touchstart','pointerdown','keydown'].forEach(type=>{
    window.addEventListener(type,()=>{
      cancelNavigation();
      keepNavigationVisible=false;
    },{passive:true});
  });
  function navigateTo(hash,{updateHistory=false,behavior='instant'}={}){
    const isTop=hash==='#'||hash==='#top';
    const section=isTop?document.body:document.getElementById(hash.slice(1));
    if(!section)return false;
    cancelNavigation();
    navigationTarget=isTop?document.body:section.querySelector('h2, h3')||section;
    keepNavigationVisible=true;
    nav.classList.remove('nav-hidden');
    activate(hash);
    if(updateHistory&&window.location.hash!==hash)history.pushState(null,'',hash);
    window.scrollTo({
      top:sectionPosition(navigationTarget),
      behavior
    });
    navigationTimer=setTimeout(finishNavigation,1200);
    return true;
  }
  document.addEventListener('click',event=>{
    const link=event.target.closest('a[href^="#"]');
    if(!link||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
    if(link.hasAttribute('download')||(link.target&&link.target!=='_self'))return;
    const behavior=window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth';
    if(navigateTo(link.getAttribute('href'),{updateHistory:true,behavior}))event.preventDefault();
  });
  window.addEventListener('load',()=>{
    if(window.location.hash)requestAnimationFrame(()=>navigateTo(window.location.hash));
  });
  window.addEventListener('hashchange',()=>{
    navigateTo(window.location.hash||'#top');
  });
  function update(){
    const position=Math.max(0,window.scrollY),delta=position-previous;
    const nextDirection=Math.sign(delta);
    if(nextDirection!==direction){travel=0;direction=nextDirection;}
    travel+=Math.abs(delta);
    if(keepNavigationVisible||position<70){nav.classList.remove('nav-hidden');travel=0;}
    else if(travel>10){
      nav.classList.toggle('nav-hidden',delta>0&&!nav.querySelector(':focus-visible'));travel=0;
    }
    previous=position;scheduled=false;
    if(!navigationTarget)updateActive();
  }
  window.addEventListener('scroll',()=>{if(!scheduled){scheduled=true;requestAnimationFrame(update);}},{passive:true});
  nav.addEventListener('focusin',()=>nav.classList.remove('nav-hidden'));
  window.addEventListener('keydown',event=>{if(event.key==='Tab')nav.classList.remove('nav-hidden');});
})();

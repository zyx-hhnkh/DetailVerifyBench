/* Local review navigation and full-table image export for the two original editions. */
(()=>{
  const controls=document.querySelector('.leader-controls');
  const exportButton=document.createElement('button');
  exportButton.className='paper-export';exportButton.textContent='Export table ↗';
  controls.append(exportButton);
  exportButton.addEventListener('click',async()=>{
    const table=document.getElementById('leader-table');
    const box=document.createElement('div');
    box.style.cssText='position:fixed;top:0;left:-15000px;padding:28px;background:#fff;color:#222';
    box.style.width=(Math.max(table.scrollWidth,1000)+56)+'px';
    box.innerHTML='<h3 style="font:500 22px Manrope;margin:0 0 12px">DetailVerifyBench</h3><p style="font:12px Manrope;margin:0 0 20px">Current manuscript · Real test · Word-level percentages · FPR and ZOR: lower is better.</p>';
    const clone=table.cloneNode(true);box.append(clone);document.body.append(box);
    clone.querySelectorAll('td,th').forEach(c=>{c.style.position='static';c.style.color='#222';c.style.background=c.tagName==='TH'?'#eee':'#fff';});
    exportButton.disabled=true;exportButton.textContent='Preparing…';
    try{
      await document.fonts.ready;
      const canvas=await html2canvas(box,{scale:2,backgroundColor:'#fff',logging:false,windowWidth:box.scrollWidth+100});
      const link=document.createElement('a');link.href=canvas.toDataURL('image/png');link.download='DetailVerifyBench_real_word_results.png';link.click();
      exportButton.textContent='Image saved';
    }catch(error){exportButton.textContent='Export unavailable';console.error('Table export failed:',error);}
    finally{box.remove();exportButton.disabled=false;setTimeout(()=>exportButton.textContent='Export table ↗',2500);}
  });
})();

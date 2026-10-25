// npm's playwright package is needed only for this browser verification.
// Start serve_web_smoke.py, then: node scripts/verify_web.cjs <screenshots-directory>
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs/promises');
const base = process.env.WEB_SMOKE_URL || 'http://127.0.0.1:8765';
const output = path.resolve(process.argv[2] || 'runs/web-smoke/screenshots');
const dataset = path.resolve(process.env.WEB_SMOKE_DATASET || 'runs/web-smoke/images.zip');
const terminal = ['completed','target_not_reached','artifact_failed','no_model','failed','cancelled','interrupted'];
(async () => {
  await fs.mkdir(output,{recursive:true});
  const browser=await chromium.launch({channel:process.env.WEB_SMOKE_BROWSER || 'msedge',headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  const errors=[];page.on('pageerror',err=>errors.push(err.message));
  async function until(check,timeout=180000) {
    const end=Date.now()+timeout;
    while(Date.now()<end) {if(await check())return;await new Promise(resolve=>setTimeout(resolve,250));}
    throw new Error('Timed out waiting for execution state');
  }
  async function layout(name) {
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${name}: horizontal overflow`);
    await page.screenshot({path:path.join(output,name+'.png'),fullPage:true});
  }
  async function snapshotBoth(name) {
    for(const width of [320,768]) {await page.setViewportSize({width,height:900});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`${name}: overflow at ${width}`);}
    await page.setViewportSize({width:1440,height:1000});await layout(name+'-desktop');
    await page.setViewportSize({width:390,height:844});await layout(name+'-mobile');
    await page.setViewportSize({width:1440,height:1000});
  }
  try {
    await page.goto(base);await page.waitForSelector('#form');
    await page.locator('.skip').focus();await page.keyboard.press('Enter');
    await page.keyboard.press('Tab');assert(await page.locator(':focus').isVisible());
    await snapshotBoth('01-start');
    await page.getByRole('button',{name:'Crear mi clasificador'}).click();
    assert(await page.locator('#file-error').isVisible());assert.equal(await page.locator(':focus').getAttribute('id'),'file');
    await page.locator('#file').setInputFiles({name:'bad.zip',mimeType:'application/zip',buffer:Buffer.from('invalid')});
    await page.getByRole('button',{name:'Crear mi clasificador'}).click();
    await page.waitForFunction(()=>!document.getElementById('go').disabled);
    assert.match(await page.locator('#file-error').textContent(),/ZIP/);
    let rid=process.env.WEB_SMOKE_RUN;
    const status=async()=> (await page.request.get(`${base}/api/runs/${rid}`)).json();
    if(!rid) {
    await page.locator('#file').setInputFiles(dataset);
    await page.locator('#use_case').fill('Distinguir formas geométricas');
    await page.locator('#budget').selectOption('rapido');
    await page.locator('.advanced summary').click();await page.locator('#pretrained').uncheck();
    await page.getByRole('button',{name:'Crear mi clasificador'}).click();
    await page.waitForFunction(()=>!document.getElementById('tracking').hidden);
    rid=(await page.locator('#run-id').textContent()).trim();
    await until(async()=>{const d=await status();return !!d.progress.current_trial || terminal.includes(d.status);});
    const before=await status();assert(before.progress.current_trial,'No active training attempt to cancel');
    await snapshotBoth('02-tracking');
    await page.locator('#cancel').click();
    await until(async()=>(await status()).status==='cancelled');
    await page.waitForFunction(()=>!document.getElementById('results').hidden,{},{timeout:15000});
    await snapshotBoth('03-cancelled');
    const cancelled=await status();assert.equal(cancelled.progress.attempts_finished,1);
    await page.getByRole('button',{name:'Seguimiento',exact:false}).first().click();
    await page.locator('#resume').click();
    await until(async()=>['completed','target_not_reached','failed','artifact_failed','no_model'].includes((await status()).status),600000);
    await page.waitForFunction(()=>!document.getElementById('results').hidden,{},{timeout:15000});
    } else {
      await page.goto(base+'/#results/'+rid);await page.waitForSelector('#results:not([hidden])');
    }
    const finished=await status();assert(['completed','target_not_reached'].includes(finished.status),JSON.stringify(finished));
    assert(finished.progress.attempts_finished>=2);assert(finished.summary.test.n_test>0);
    assert.equal(finished.summary.verification.checks.local.status,'passed');
    for(const url of Object.values(finished.downloads)) assert.equal((await page.request.get(base+url)).status(),200);
    await page.locator('#report-details summary').click();await page.waitForFunction(()=>document.getElementById('report').textContent.includes('#'));
    await snapshotBoth('04-results');
    // Temporary loss of API access must retain the project and recover.
    await page.route('**/api/runs/'+rid,route=>route.abort());
    await page.reload();await page.waitForSelector('#connection:not([hidden])');
    assert((await page.evaluate(()=>localStorage.getItem('automl.run')))===rid);
    await page.unroute('**/api/runs/'+rid);await page.waitForSelector('#results:not([hidden])',{timeout:20000});
    // Restore results directly from the stored reference and server state.
    await page.goto(base+'/#results/'+rid);await page.waitForSelector('#results:not([hidden])');
    assert.equal((await page.locator('#run-id').textContent()).trim(),rid);
    // Exercise a failed delivery in the browser using a clearly simulated response.
    const partial={...finished,status:'artifact_failed',capabilities:{cancel:false,resume:false},
      summary:{...finished.summary,status:'artifact_failed',reason:'packaging_failed',
        error:{message:'Prueba de interfaz: falló la preparación de la API.'},verification:{status:'failed',verified:false}},
      stop_reason:'packaging_failed',downloads:Object.fromEntries(Object.entries(finished.downloads).filter(([kind])=>kind!=='artifact'))};
    await page.route('**/api/runs/'+rid,route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(partial)}));
    await page.reload();await page.waitForSelector('#run-error:not([hidden])');
    assert.equal(await page.locator('#downloads a[href*="kind=artifact"]').count(),0);
    assert.equal(await page.locator('#result-resume').isVisible(),false);
    await snapshotBoth('05-partial-simulated');
    await page.unroute('**/api/runs/'+rid);await page.reload();await page.waitForSelector('#results:not([hidden])');
    assert.deepEqual(errors,[]);
    await fs.writeFile(path.join(output,'verification.json'),JSON.stringify({run_id:rid,status:finished.status,attempts:finished.progress,verification:finished.summary.verification,viewports:[1440,390,320,768],partial_delivery_ui:'simulated',browser_errors:errors},null,2));
    console.log(JSON.stringify({run_id:rid,status:finished.status,screenshots:output,browser_errors:errors}));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

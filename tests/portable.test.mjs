import test from 'node:test';
import assert from 'node:assert/strict';
import {aqua_create_library} from '../runtimes/portable/opencv.mjs';
const encoder=new TextEncoder(),decoder=new TextDecoder();
test('portable OpenCV executes every advertised image operation and releases handles',async()=>{
 const api=aqua_create_library();
 const request=async q=>JSON.parse(decoder.decode(await api.cv_command(encoder.encode(JSON.stringify(q)))));
 try {
  const sid=(await request({op:'open'})).value;
  const command=q=>request({...q,session:sid});
  const capabilities=(await command({op:'info'})).value;
  assert.match(capabilities.version,/^4\./);
  const input=(await command({op:'sample',frame:10})).value;
  const failures=[];
  for(const feature of capabilities.features.filter(x=>x!=='face')){
   const result=await command({op:'apply',image:input,feature,parameters:{}});
   if(!result.ok){failures.push(feature+': '+result.error);continue;}
   const dimensions=await command({op:'inspect',image:result.value.image});assert.ok(dimensions.value.width>0);
   await command({op:'release',image:result.value.image});
  }
  assert.deepEqual(failures,[]);
  const imported=(await command({op:'rgba',width:2,height:1})).value;
  const header=(...values)=>{const out=new Uint8Array(values.length*4);values.forEach((v,i)=>new DataView(out.buffer).setUint32(i*4,v,true));return out;};
  const rgba=Uint8Array.of(255,0,0,255,0,128,255,255),transfer=new Uint8Array(12+rgba.length);transfer.set(header(sid,imported,0));transfer.set(rgba,12);
  await api.cv_write(transfer);assert.deepEqual(await api.cv_pixels(header(sid,imported,0,8)),rgba);
  await assert.rejects(api.cv_pixels(header(sid,imported,0,12)),/Invalid RGBA chunk/);
  await command({op:'release',image:imported});
  await command({op:'release',image:input});assert.equal((await command({op:'info'})).value.images,0);
  assert.equal((await command({op:'inspect',image:input})).ok,false);
  await command({op:'close'});assert.equal((await command({op:'sample'})).ok,false);
 }finally{api.aqua_dispose();api.aqua_dispose();}
});

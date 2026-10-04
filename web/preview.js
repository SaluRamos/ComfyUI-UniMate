import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
app.registerExtension({name:'UniMate.AnimationPreview',beforeRegisterNodeDef(nodeType,nodeData){
if(nodeData.name!=='UniMateAnimate')return;
const created=nodeType.prototype.onNodeCreated,executed=nodeType.prototype.onExecuted,removed=nodeType.prototype.onRemoved;
nodeType.prototype.onNodeCreated=function(){const result=created?.apply(this,arguments);
const container=document.createElement('div');container.style.cssText='width:100%;height:420px;display:flex;flex-direction:column';
const choice=document.createElement('select');choice.title='Animação gerada';choice.style.cssText='background:#333;color:white;width:100%';container.appendChild(choice);
const iframe=document.createElement('iframe');iframe.src=new URL('./player.html',import.meta.url).href;iframe.style.cssText='border:0;flex:1;width:100%;min-height:0';container.appendChild(iframe);
let files=[],ready=false;const send=()=>{const f=files[Number(choice.value)||0];if(ready&&f)iframe.contentWindow.postMessage({type:'unimate-load',url:api.apiURL('/view?'+new URLSearchParams(f))},location.origin);};
const listener=e=>{if(e.source===iframe.contentWindow&&e.origin===location.origin&&e.data?.type==='unimate-ready'){ready=true;send();}};window.addEventListener('message',listener);choice.onchange=send;
this._unimatePreview={load(items){files=items;choice.replaceChildren(...items.map((f,i)=>{const option=document.createElement('option');option.value=i;option.textContent=f.filename;return option;}));send();},dispose(){window.removeEventListener('message',listener);iframe.remove();}};
this.addDOMWidget('animation_preview','unimate_preview',container,{serialize:false,getMinHeight:()=>420,getMaxHeight:()=>420});this.setSize([Math.max(this.size[0],560),this.size[1]]);return result;};
nodeType.prototype.onExecuted=function(message){const result=executed?.apply(this,arguments);this._unimatePreview?.load(message.unimate_animation??[]);return result;};
nodeType.prototype.onRemoved=function(){this._unimatePreview?.dispose();return removed?.apply(this,arguments);};
}});

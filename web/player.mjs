/* Three.js runs in this iframe, isolated from other ComfyUI extensions. */
const view = document.getElementById('view'), status = document.getElementById('status');
const play = document.getElementById('play'), time = document.getElementById('time'), label = document.getElementById('label'), speed = document.getElementById('speed');
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x242424);
const camera = new THREE.PerspectiveCamera(45, 1, .001, 10000);
const renderer = new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); view.appendChild(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x666666,1.5));
const light = new THREE.DirectionalLight(0xffffff,1.5); light.position.set(3,5,4); scene.add(light);
let model=null, mixer=null, duration=0, playing=false, generation=0, frame=0;
let target=new THREE.Vector3(), radius=3, azimuth=.5, polar=1.3;
function positionCamera(){camera.position.set(target.x+radius*Math.sin(polar)*Math.sin(azimuth),target.y+radius*Math.cos(polar),target.z+radius*Math.sin(polar)*Math.cos(azimuth));camera.lookAt(target);}
function dispose(root){if(!root)return;root.traverse(o=>{o.geometry?.dispose();for(const m of (Array.isArray(o.material)?o.material:[o.material])){if(!m)continue;for(const v of Object.values(m))if(v?.isTexture)v.dispose();m.dispose();}});}
function setPlaying(value){playing=value;play.textContent=playing?'❚❚':'▶';}
function updateLabel(){const t=mixer?.time??0;time.value=t;label.textContent=`${t.toFixed(2)} / ${duration.toFixed(2)} s`;}
async function load(url){const token=++generation;status.textContent='Carregando animação…';setPlaying(false);play.disabled=time.disabled=true;
try{const gltf=await new THREE.GLTFLoader().loadAsync(url);if(token!==generation){dispose(gltf.scene);return;}
if(mixer){mixer.stopAllAction();mixer.uncacheRoot(model);}scene.remove(model);dispose(model);model=gltf.scene;scene.add(model);
const bounds=new THREE.Box3().setFromObject(model),size=bounds.getSize(new THREE.Vector3());bounds.getCenter(target);radius=Math.max(size.length()*1.15,.1);camera.near=radius/1000;camera.far=radius*100;camera.updateProjectionMatrix();positionCamera();
mixer=new THREE.AnimationMixer(model);duration=Math.max(0,...gltf.animations.map(c=>c.duration));
// Each exported file contains one generated action. Avoid blending duplicate tracks.
if(gltf.animations.length){mixer.clipAction(gltf.animations[0]).play();
const animatedBounds=new THREE.Box3();
for(let i=0;i<=20;i++){mixer.setTime(duration*i/20);model.updateMatrixWorld(true);model.traverse(o=>{if(o.isSkinnedMesh){const point=new THREE.Vector3();for(let v=0;v<o.geometry.attributes.position.count;v+=Math.max(1,Math.floor(o.geometry.attributes.position.count/10000))){point.fromBufferAttribute(o.geometry.attributes.position,v);o.boneTransform(v,point);point.applyMatrix4(o.matrixWorld);animatedBounds.expandByPoint(point);}}else if(o.isMesh)animatedBounds.expandByObject(o);});}
if(!animatedBounds.isEmpty()){animatedBounds.getCenter(target);radius=Math.max(animatedBounds.getSize(new THREE.Vector3()).length()*1.15,.1);camera.far=radius*100;camera.updateProjectionMatrix();positionCamera();}mixer.setTime(0);
}
time.max=duration||1;play.disabled=time.disabled=duration===0;status.textContent=duration?'Arraste para girar · roda para zoom':'Arquivo sem animação';setPlaying(duration>0);updateLabel();
}catch(e){status.textContent=`Não foi possível visualizar: ${e.message}`;}}
play.onclick=()=>setPlaying(!playing);time.oninput=()=>{if(mixer){mixer.setTime(Number(time.value));updateLabel();}};
let drag=null;view.onpointerdown=e=>{drag=[e.clientX,e.clientY];view.setPointerCapture(e.pointerId);};view.onpointerup=view.onpointercancel=()=>drag=null;
view.onpointermove=e=>{if(!drag)return;azimuth-=(e.clientX-drag[0])*.01;polar=Math.max(.05,Math.min(Math.PI-.05,polar+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];positionCamera();};
view.onwheel=e=>{e.preventDefault();radius*=Math.exp(e.deltaY*.001);positionCamera();};
const observer=new ResizeObserver(()=>{const w=view.clientWidth,h=view.clientHeight;if(w&&h){renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}});observer.observe(view);
const clock=new THREE.Clock();function tick(){frame=requestAnimationFrame(tick);const delta=Math.min(clock.getDelta(),.1);if(mixer&&playing&&duration){mixer.setTime((mixer.time+delta*Number(speed.value))%duration);updateLabel();}renderer.render(scene,camera);}tick();
window.addEventListener('message',e=>{if(e.origin!==location.origin||e.source!==parent)return;if(e.data?.type==='unimate-load')load(e.data.url);});
window.addEventListener('pagehide',()=>{generation++;cancelAnimationFrame(frame);observer.disconnect();mixer?.stopAllAction();dispose(model);renderer.dispose();});
parent.postMessage({type:'unimate-ready'},location.origin);

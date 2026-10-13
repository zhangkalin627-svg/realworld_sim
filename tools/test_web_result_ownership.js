// Public UI functions executed with a small DOM/fetch boundary.
const vm=require('node:vm'), fs=require('node:fs'), assert=require('node:assert/strict');
const nodes={};
const make=()=>({textContent:'',innerHTML:'',style:{},disabled:false,appendChild(){},querySelectorAll(){return []},classList:{toggle(){}}});
const document={querySelector:s=>nodes[s]||(nodes[s]=make()),querySelectorAll:()=>[],createElement:make};
const ctx=vm.createContext({document,window:{},console,setTimeout,clearTimeout,FormData:class{},fetch:async()=>({json:async()=>({jobs:[]})})});
let src=fs.readFileSync('web/static/app.js','utf8').replace(/boot\(\);\s*$/,'');
vm.runInContext(src,ctx);
vm.runInContext("MODULES=[{key:'a',name:'Module A',fields:[]},{key:'b',name:'Module B',fields:[]}];select(0);",ctx);
vm.runInContext("renderResult({module:'a',status:'done',result:{lines:['A output']}})",ctx);
assert.match(nodes['#status'].textContent,/Module A/);
vm.runInContext('select(1)',ctx);
assert.equal(nodes['#lines'].textContent,'');
vm.runInContext("renderResult({module:'a',status:'done',result:{lines:['late A']}})",ctx);
assert.equal(nodes['#lines'].textContent,'');
vm.runInContext("renderResult({module:'b',status:'done',result:{lines:['B output']}})",ctx);
assert.match(nodes['#status'].textContent,/Module B/);
assert.equal(nodes['#lines'].textContent,'B output');
console.log('UI result ownership: module label, switch clearing, stale reject, current render passed');

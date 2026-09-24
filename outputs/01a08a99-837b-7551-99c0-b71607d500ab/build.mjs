import fs from 'node:fs/promises';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const out = new URL('.',import.meta.url).pathname;
const wb=Workbook.create();
function sheet(name,headers,rows,widths){
 const s=wb.worksheets.add(name);s.showGridLines=false;
 s.getRangeByIndexes(0,0,1,headers.length).values=[headers];
 s.getRangeByIndexes(1,0,rows.length,headers.length).values=rows;
 const all=s.getRangeByIndexes(0,0,rows.length+1,headers.length);
 all.format.font={name:'Helvetica Neue',size:11,color:'#243247'};all.format.wrapText=true;all.format.verticalAlignment='center';
 s.getRangeByIndexes(0,0,1,headers.length).format={fill:'#243B53',font:{name:'Helvetica Neue',size:11,bold:true,color:'#FFFFFF'},rowHeight:42};
 widths.forEach((w,i)=>s.getRangeByIndexes(0,i,rows.length+1,1).format.columnWidth=w);
 for(let r=1;r<=rows.length;r++){s.getRangeByIndexes(r,0,1,headers.length).format.rowHeight=64;if(r%2===0)s.getRangeByIndexes(r,0,1,headers.length).format.fill='#F1F5F9';}
 return s;
}
const sets=[['DLRSD',17,17,7002,5601,1401,'Full evaluation includes the training split.'],['iSAID',15,15,24439,18076,6363,'Full evaluation includes the training split.'],['Potsdam',6,6,20102,null,null,'All registered evaluation samples.'],['Vaihingen',6,6,2254,null,null,'All registered evaluation samples.'],['UDD5',5,5,160,null,null,'Evaluation combines 120 train and 40 validation images. Not a source training dataset in this protocol.'],['LoveDA',8,7,2522,null,null,'Registered labels include no-data. Coverage excludes no-data and retains Background.'],['UAVid',8,8,270,null,null,'All registered evaluation samples.'],['VDD',7,7,400,null,null,'All registered evaluation samples.']];
const overview=sheet('Dataset Summary',['Dataset','Registered classes','Coverage classes','Evaluation samples','Training samples','Validation samples','Remarks'],sets,[18,17,17,19,19,19,66]);overview.getRange('B2:F9').setNumberFormat('#,##0');
const notes=[
 ['Scope','OVSISBench version used by PCA-Seg. Samples are loaded images or patches, not original source scenes. Counts reflect saved September 2026 run logs.'],
 ['Known classes','Known = Strong coverage + Semantic coverage. Count each target class once, even when multiple source classes map to it.'],
 ['Strong coverage','Exact labels, singular/plural variants, or clear synonyms. Examples: buildings / Building, airplane / plane, tanks / storage tank.'],
 ['Semantic coverage','Related labels, broader/narrower categories, or part/whole relations. These are counted as known under the requested inclusive convention, without claiming equivalent annotation boundaries.'],
 ['Mapping scope','Mappings extend the preceding conservative comparison by including its semantically related classes. They are analyst-defined label mappings, not an official benchmark known/unknown split.'],
 ['Denominator','Coverage classes include background, clutter and other. LoveDA no-data is excluded. These residual categories stay unknown because the source datasets have no equivalent named class.'],
 ['Source: labels','register_ovsisbench.py: DLRSD_CLASSES and ISAID_CLASSES. register_ovsisbench_eval.py: DATASET_CATEGORIES and evaluation registrations.'],
 ['Source: evaluation counts','Run rskt_clipl_all8_20260909, per-dataset log.txt, Loaded images with semantic segmentation entries. UDD5 combines its two entries.'],
 ['Source: train/validation counts','Run base_content1_context0p1_smoke_20260910, DLRSD and iSAID log.txt. Loaded entries: 5,601 / 1,401 and 18,076 / 6,363.'],
 ['Blank split cells','A blank means that no source-training split is reported for this experimental protocol. It does not mean zero samples.']
];
overview.getRange('A12').values=[['Definitions and sources']];overview.getRange('A12').format.font={bold:true,size:13};
notes.forEach(([a,b],i)=>{let r=i+13;overview.getRange(`A${r}:B${r}`).merge();overview.getRange(`A${r}`).values=[[a]];overview.getRange(`C${r}:G${r}`).merge();overview.getRange(`C${r}`).values=[[b]];overview.getRange(`A${r}:G${r}`).format={font:{name:'Helvetica Neue',size:11},wrapText:true,rowHeight:46,verticalAlignment:'center'};});
const dl=[
 ['iSAID',['ship','plane','storage tank'],['tennis court','basketball court','large vehicle','small vehicle','harbor'],'plane ~ airplane; storage tank ~ tanks','tennis court / basketball court ~ court; large vehicle / small vehicle ~ cars (vehicle family); harbor ~ dock (related port structures)'],
 ['Potsdam',['Building','Tree','Car'],['Impervious surfaces','Low vegetation'],'Building ~ buildings; Tree ~ trees; Car ~ cars','Impervious surfaces ~ pavement; Low vegetation ~ grass'],
 ['Vaihingen',['Building','Tree','Car'],['Impervious surfaces','Low vegetation'],'Building ~ buildings; Tree ~ trees; Car ~ cars','Impervious surfaces ~ pavement; Low vegetation ~ grass'],
 ['UDD5',['Building'],['Vegetation','Road','Vehicle'],'Building ~ buildings','Vegetation ~ trees / grass / chaparral; Road ~ pavement; Vehicle ~ cars'],
 ['LoveDA',['Building','Water'],['Road','Barren','Forest','Agriculture'],'Building ~ buildings; Water ~ water','Road ~ pavement; Barren ~ bare soil / sand; Forest ~ trees; Agriculture ~ field'],
 ['UAVid',['Building','Tree'],['Road','Low vegetation','Moving car','Static car'],'Building ~ buildings; Tree ~ trees','Road ~ pavement; Low vegetation ~ grass; Moving car / Static car ~ cars'],
 ['VDD',['water'],['wall','road','vegetation','vehicle','roof'],'water ~ water','wall / roof ~ buildings (parts); road ~ pavement; vegetation ~ trees / grass / chaparral; vehicle ~ cars']
];
const isa=[
 ['DLRSD',['airplane','ship','tanks'],['cars','court','dock'],'airplane ~ plane; ship ~ ship; tanks ~ storage tank','cars ~ small vehicle; court ~ tennis court / basketball court; dock ~ harbor'],
 ['Potsdam',[],['Car'],'None','Car ~ small vehicle'],['Vaihingen',[],['Car'],'None','Car ~ small vehicle'],
 ['UDD5',[],['Vehicle'],'None','Vehicle ~ large vehicle / small vehicle'],['LoveDA',[],[],'None','None'],
 ['UAVid',[],['Moving car','Static car'],'None','Moving car / Static car ~ small vehicle'],['VDD',[],['vehicle'],'None','vehicle ~ large vehicle / small vehicle']
];
const classNames={};for(const [d] of sets){classNames[d]=JSON.parse(await fs.readFile(new URL(`../../datasets/${d==='UAVid'?'uavid':d}.json`,import.meta.url),'utf8')).filter(x=>x!=='no-data');}
for(const [train,data] of [['DLRSD',dl],['iSAID',isa]]){
 const rows=data.map(([d,strong,sem,sm,rm])=>{
  const classes=classNames[d];const known=[...strong,...sem];if(new Set(known).size!==known.length||known.some(c=>!classes.includes(c)))throw Error('Invalid mapping '+d);
  return [d,classes.length,strong.length,sem.length,null,null,null,strong.join(', ')||'None',sem.join(', ')||'None',classes.filter(c=>!known.includes(c)).join(', ')||'None',`Strong coverage: ${sm}.\nSemantic coverage: ${rm}.`];
 });
 const s=sheet(`${train} Coverage`,['Target dataset','Target classes','Strong classes','Semantic classes','Known classes','Unknown classes','Known coverage','Strong coverage classes','Semantic coverage classes','Unknown classes (names)','Remarks: target ~ training class'],rows,[17,12,12,13,12,13,14,29,37,35,80]);
 for(let r=2;r<=8;r++){s.getRange(`E${r}:G${r}`).formulas=[[`=C${r}+D${r}`,`=B${r}-E${r}`,`=E${r}/B${r}`]];s.getRange(`A${r}:K${r}`).format.rowHeight=105;}
 s.getRange('B2:F8').setNumberFormat('0');s.getRange('G2:G8').setNumberFormat('0.0%');
 s.getRange('C2:C8').format.fill='#E2F0E9';s.getRange('D2:D8').format.fill='#FFF1D6';
 s.getRange('A10').values=[['Known = Strong + Semantic. LoveDA excludes no-data. See Dataset Summary for definitions and sources.']];
 s.getRange('A10:K10').merge();s.getRange('A10:K10').format={font:{name:'Helvetica Neue',size:11},rowHeight:30};
 console.log((await wb.inspect({kind:'table',range:`'${train} Coverage'!A1:G8`,include:'values,formulas',tableMaxRows:8,tableMaxCols:7})).ndjson);
}
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!',options:{useRegex:true,maxResults:20},summary:'Formula error scan'})).ndjson);
for(const name of ['Dataset Summary','DLRSD Coverage','iSAID Coverage']){const p=await wb.render({sheetName:name,range:name==='Dataset Summary'?'A1:G22':'A1:K10',scale:1,format:'png'});await fs.writeFile(out+name+'.png',new Uint8Array(await p.arrayBuffer()));}
await (await SpreadsheetFile.exportXlsx(wb)).save(out+'Remote_Sensing_Dataset_Coverage.xlsx');

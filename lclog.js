/*
copyright louis chevallier
*/


EKOTable = {};

/*

100 : a=23;
101 : EKOX(a);

will print :

yourfile.js:101 : a=23

*/

function LOG(...txt) {
    const s = LOGS(2, ...txt);
    console.log(s);
}

function EKOT(...txt) {
    const s = LOGS(2, ...txt);    
    console.log(s);    
    try {
        const xhr1 = new XMLHttpRequest();
        xhr1.open("GET", "log?data=" + s);
        xhr1.send();
    } catch (err) {
    }
}

function EKOX(...txt) {
    const s = LOGS(2, ...txt);    
    console.log(s);    
    try {
        const xhr1 = new XMLHttpRequest();
        xhr1.open("GET", "log?data=" + s);
        xhr1.send();
    } catch (err) {
    }
}

function LOGS(level=1, ...txt) {
    const thisline = new Error().lineNumber
    const error = new Error();
    const stack = error.stack.split('\n')
    //const level = 2;
    const a = stack[level].split("@")
    const b = a[1].split(":")
    const nl = parseInt(b.slice(-2, -1));
    const url = b.slice(0, -2).join(":")
    const file = url;
    if (! EKOTable.hasOwnProperty(file)) {
        const req = new XMLHttpRequest();
        req.open("GET", url, false); // <-- completely sync and deprecated
        req.send();
        if(req.readyState === 4 && req.status === 200) {
            //console.log("response=" + req.response);
            const t  = req.response;
            const l = t.split("\n");
            EKOTable[file] = l;
        } else {
            console.log("unable to retreive " + file);
        }
    } else {
        //console.log("already there");
    }
    const l = EKOTable[file];
    const vv = l[nl-1];
    const re = new RegExp(" *EKOX?\\(([^\\)]+)\\) *;?");
    const result = re.exec(vv);
    const vvn1 = result[1];
    const vvn = vvn1.split(',');
    var s = ""
    for (i in txt) {
        s +=  (i>0 ? ", " : "") + vvn[i] + "=" + txt[i] + " ";
    }
    const r = file + ":" + nl + ":" + s;
    return r;
}






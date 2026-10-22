const tabs=[...document.querySelectorAll(".tab")];
const pages=[...document.querySelectorAll(".page")];
tabs.forEach(btn=>btn.addEventListener("click",()=>{
  tabs.forEach(t=>t.classList.toggle("active",t===btn));
  pages.forEach(p=>p.classList.toggle("active",p.id===btn.dataset.page));
}));

async function loadCSV(path){
  const response=await fetch(path);
  if(!response.ok) throw new Error("Could not load "+path);
  const text=await response.text();
  const rows=text.trim().split(/\r?\n/).map(line=>line.split(","));
  return rows;
}
function renderTable(targetId, rows){
  const target=document.getElementById(targetId);
  if(!rows.length){target.textContent="No results available.";return;}
  const table=document.createElement("table");
  rows.forEach((row,i)=>{
    const tr=document.createElement("tr");
    row.forEach(value=>{
      const cell=document.createElement(i===0?"th":"td");
      cell.textContent=value;
      tr.appendChild(cell);
    });
    table.appendChild(tr);
  });
  target.replaceChildren(table);
}
(async()=>{
  try{
    const mapping=await loadCSV("results/missingness_noise_mapping.csv");
    renderTable("mappingTable",mapping);
    const hospital=await loadCSV("results/hospital_wise_results.csv");
    renderTable("hospitalTable",hospital);
    const metrics=await loadCSV("results/final_model_results.csv");
    renderTable("metricsTable",metrics);
    const mia=await loadCSV("results/mia_privacy_results.csv");
    renderTable("miaTable",mia);
  }catch(error){
    console.error(error);
    document.getElementById("mappingTable").textContent=
      "Open this dashboard through the Python backend to load its data files.";
  }
})();

document.getElementById("checkBackend").addEventListener("click",async()=>{
  const status=document.getElementById("backendStatus");
  try{
    const response=await fetch("/api/health");
    if(!response.ok) throw new Error("Backend unavailable");
    const data=await response.json();
    status.textContent="Backend status: "+data.status;
  }catch(error){
    status.textContent="Backend status: unavailable. Start the Python backend first.";
  }
});

const featureNames=[
  "age","al","appet","ba","bgr","bp","bu","hemo","pc","pcc",
  "pcv","pot","rbc","rbcc","sc","sg","sod","su","wbcc"
];
const numericFeatures=new Set([
  "age","al","bgr","bp","bu","hemo","pcv","pot","rbcc","sc","sg","sod","su","wbcc"
]);
const categoryOptions={
  appet:["good","poor"],ba:["notpresent","present"],
  pc:["normal","abnormal"],pcc:["notpresent","present"],
  rbc:["normal","abnormal"]
};
const inputArea=document.getElementById("featureInputs");
featureNames.forEach(name=>{
  const label=document.createElement("label");
  label.append(document.createTextNode(name.toUpperCase()));
  let input;
  if(categoryOptions[name]){
    input=document.createElement("select");
    const empty=document.createElement("option");
    empty.value="";empty.textContent="Select";input.append(empty);
    categoryOptions[name].forEach(option=>{
      const opt=document.createElement("option");
      opt.value=option;opt.textContent=option;input.append(opt);
    });
  }else{
    input=document.createElement("input");
    input.type="number";input.step="any";
    input.placeholder="Enter value";
  }
  input.name=name;
  input.required=true;
  label.append(input);inputArea.append(label);
});
document.getElementById("predictionForm").addEventListener("submit",async event=>{
  event.preventDefault();
  const result=document.getElementById("predictionResult");
  result.textContent="Sending prediction request...";
  const features={};
  new FormData(event.currentTarget).forEach((value,key)=>features[key]=value);
  try{
    const response=await fetch("/api/predict",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({features})
    });
    const data=await response.json();
    if(!response.ok) throw new Error(data.error||"Prediction failed");
    result.replaceChildren();

    const resultText = document.createElement("p");
    resultText.textContent = "Research output: " + data.prediction +
      " | Probability: " + Number(data.probability).toFixed(4) +
      " | Not a clinical diagnosis.";
    result.appendChild(resultText);

    const pdfButton = document.createElement("button");
    pdfButton.type = "button";
    pdfButton.className = "primary";
    pdfButton.textContent = "Download Patient Report PDF";

    pdfButton.addEventListener("click", async () => {
      pdfButton.disabled = true;
      pdfButton.textContent = "Generating PDF...";

      try {
        const formData = new FormData(
          document.getElementById("predictionForm")
        );
        const featuresForReport = {};
        formData.forEach((value, key) => {
          featuresForReport[key] = value;
        });

        const patientId =
          document.getElementById("patientId")?.value || "Anonymous";
        const hospital =
          document.getElementById("hospitalSelect")?.value || 1;

        await window.downloadPatientReport(
          patientId,
          hospital,
          featuresForReport,
          data
        );
      } catch (error) {
        alert("PDF download failed: " + error.message);
      } finally {
        pdfButton.disabled = false;
        pdfButton.textContent = "Download Patient Report PDF";
      }
    });

    result.appendChild(pdfButton);
  }catch(error){
    result.textContent="Prediction unavailable: "+error.message+
      ". Backend/model compatibility must be checked first.";
  }
});



// PDF download helper
window.downloadPatientReport = async function(
    patientId,
    hospital,
    features,
    result
) {
    const response = await fetch("/api/patient-report", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            patient_id: patientId,
            hospital: hospital,
            features: features,
            prediction: result.prediction ?? result.label ?? "Not provided",
            probability: result.probability ?? result.confidence ?? null
        })
    });

    if (!response.ok) {
        const message = await response.text();
        throw new Error("PDF API error (" + response.status + "): " + message);
    }

    const blob = await response.blob();

    if (blob.type && !blob.type.includes("pdf")) {
        throw new Error("Server response is not a PDF.");
    }

    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = downloadUrl;
    link.download = "ckd_patient_report.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(downloadUrl);
};




// PDF download helper
window.downloadPatientReport = async function(
    patientId,
    hospital,
    features,
    result
) {
    const response = await fetch("/api/patient-report", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            patient_id: patientId,
            hospital: hospital,
            features: features,
            prediction: result.prediction ?? result.label ?? "Not provided",
            probability: result.probability ?? result.confidence ?? null
        })
    });

    if (!response.ok) {
        const message = await response.text();
        throw new Error("PDF API error (" + response.status + "): " + message);
    }

    const blob = await response.blob();

    if (blob.type && !blob.type.includes("pdf")) {
        throw new Error("Server response is not a PDF.");
    }

    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = downloadUrl;
    link.download = "ckd_patient_report.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(downloadUrl);
};


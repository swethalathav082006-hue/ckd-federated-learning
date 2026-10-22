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

let latestPrediction = null;

document.getElementById("predictionForm").addEventListener("submit", async event => {
  event.preventDefault();

  const result = document.getElementById("predictionResult");
  result.replaceChildren();
  result.textContent = "Running research prediction...";

  const patientId = document.getElementById("patientId").value.trim();
  const hospital = Number(document.getElementById("hospitalSelect").value);
  const sex = document.getElementById("patientSex").value;

  if (!patientId) {
    result.textContent = "Please enter a Patient ID.";
    return;
  }

  // Send only the 19 model features to the prediction endpoint.
  const formData = new FormData(event.currentTarget);
  const features = {};

  featureNames.forEach(name => {
    features[name] = formData.get(name);
  });

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({hospital, features})
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    latestPrediction = {
      patient_id: patientId,
      hospital,
      sex,
      features,
      prediction: data.prediction,
      probability: data.probability,
      threshold: data.threshold
    };

    const summary = document.createElement("p");
    summary.textContent =
      "Patient ID: " + patientId +
      " | Hospital " + hospital +
      " | Result: " + data.prediction +
      " | Probability: " + Number(data.probability).toFixed(4) +
      " | Research demonstration only; not a medical diagnosis.";

    result.replaceChildren(summary);

    const warning = document.createElement("p");
    warning.className = "warning";
    warning.textContent = data.warning ||
      "Research prototype only. Not for clinical decision-making.";
    result.appendChild(warning);

    const pdfButton = document.createElement("button");
    pdfButton.type = "button";
    pdfButton.className = "primary";
    pdfButton.textContent = "Download Patient PDF Report";

    pdfButton.addEventListener("click", async () => {
      pdfButton.disabled = true;
      pdfButton.textContent = "Generating PDF...";

      try {
        const reportFeatures = {
          ...latestPrediction.features,
          "Sex (not used by model)": latestPrediction.sex || "Not specified"
        };

        const pdfResponse = await fetch("/api/patient-report", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            patient_id: latestPrediction.patient_id,
            hospital: latestPrediction.hospital,
            features: reportFeatures,
            prediction: latestPrediction.prediction,
            probability: latestPrediction.probability,
            threshold: latestPrediction.threshold
          })
        });

        if (!pdfResponse.ok) {
          let message = "PDF generation failed.";
          try {
            const errorData = await pdfResponse.json();
            message = errorData.error || message;
          } catch (_) {}
          throw new Error(message);
        }

        const blob = await pdfResponse.blob();
        if (blob.type && !blob.type.includes("application/pdf")) {
          throw new Error("The server did not return a PDF file.");
        }

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "CKD_Report_" +
          latestPrediction.patient_id.replace(/[^a-zA-Z0-9_-]/g, "_") +
          ".pdf";
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);

        pdfButton.textContent = "Download Patient PDF Report";
      } catch (error) {
        alert("PDF download failed: " + error.message);
        pdfButton.textContent = "Retry PDF Download";
      } finally {
        pdfButton.disabled = false;
      }
    });

    result.appendChild(pdfButton);

  } catch (error) {
    latestPrediction = null;
    result.textContent = "Prediction unavailable: " + error.message;
  }
});

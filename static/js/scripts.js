document.getElementById("dataForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch("/", {
        method: "POST",
        headers: {
            "X-Requested-With": "XMLHttpRequest"  // Add this header
        },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("plotContainer").innerHTML = `
            <h2>Generated Data</h2>
            <img src="${data.plot1_url}" alt="Scatter Plot with Regression Line" style="width: 600px;">
            <img src="${data.plot2_url}" alt="Histogram of Slopes and Intercepts" style="width: 600px;">
            <p>Proportion of slopes more extreme: ${(data.slope_extreme * 100).toFixed(2)}%</p>
            <p>Proportion of intercepts more extreme: ${(data.intercept_extreme * 100).toFixed(2)}%</p>
        `;
        
        document.getElementById("hypothesisTestSection").style.display = "block";
        document.getElementById("confidenceIntervalSection").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
});


// Hypothesis Testing
document.getElementById("hypothesisForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch("/hypothesis_test", {
        method: "POST",
        headers: {
            "X-Requested-With": "XMLHttpRequest"  // Ensure this is set to signal an AJAX request
        },
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("hypothesisResults").innerHTML = `
            <p>p-value: ${data.p_value.toFixed(4)}</p>
            ${data.fun_message ? `<p>${data.fun_message}</p>` : ""}
            <img src="${data.plot3_url}" alt="Hypothesis Testing Histogram" style="width: 600px;">
        `;
    })
    .catch(error => console.error("Error:", error));
});


// Confidence Interval Calculation
document.getElementById("confidenceForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch("/confidence_interval", {
        method: "POST",
        body: formData,
        "X-Requested-With": "XMLHttpRequest"
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok " + response.statusText);
        }
        return response.json(); // This will throw an error if the response is not valid JSON
    })
    .then(data => {
        document.getElementById("confidenceResults").innerHTML = `
            <p>Confidence Interval for ${data.parameter}: [${data.ci_lower.toFixed(2)}, ${data.ci_upper.toFixed(2)}]</p>
            <p>Includes true parameter: ${data.includes_true ? "Yes" : "No"}</p>
            <img src="${data.plot4_url}" alt="Confidence Interval Plot" style="width: 600px;">
        `;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("confidenceResults").innerHTML = `
            <p style="color: red;">An error occurred: ${error.message}</p>
        `;
    });
});
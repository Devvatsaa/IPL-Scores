function predictScore() {
    // Retrieve form data
    var form = document.getElementById("prediction-form");
    var formData = new FormData(form);
  
    // Send form data to the backend using AJAX
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/predict", true);
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xhr.onreadystatechange = function () {
      if (xhr.readyState === 4 && xhr.status === 200) {
        // Update UI with prediction result
        document.getElementById("result").innerText = "Predicted Score: " + xhr.responseText;
      }
    };
    xhr.send(new URLSearchParams(formData).toString());
  }
  
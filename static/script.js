function submitForm() {
    var form = document.getElementById("predict-form");
    var formData = new FormData(form);
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/predict");
    xhr.onload = function() {
      if (xhr.status === 200) {
        var resultDiv = document.getElementById("result");
        // We will recieve object as a string, so we need to parse it to JSON
        var objectReceived = JSON.parse(xhr.responseText);

        var answer = objectReceived.answer;
        var answer_type = objectReceived.answer_type;
        var answerability = objectReceived.answerability;
        console.log(answer);

        resultDiv.innerHTML = "<p>Answer: " + answer + "</p>" +
                                "<p>Answer type: " + answer_type + "</p>" +
                                "<p>Answerability(The confidence for being unanswerable): " + answerability + "</p>";
        } else {
        alert("An error occurred while trying to predict the answer.");
      }
    };
    xhr.send(formData);
  }
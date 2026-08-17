/* =====================================================
   CARDIOPREDICT FRONTEND
===================================================== */


/* -----------------------------------------------------
   ELEMENTS
----------------------------------------------------- */

const form = document.getElementById("predictionForm");

const predictButton =
    document.getElementById("predictButton");

const emptyState =
    document.getElementById("emptyState");

const resultContent =
    document.getElementById("resultContent");

const riskPercentage =
    document.getElementById("riskPercentage");

const riskBadge =
    document.getElementById("riskBadge");

const riskTitle =
    document.getElementById("riskTitle");

const riskDescription =
    document.getElementById("riskDescription");

const diseaseProbability =
    document.getElementById("diseaseProbability");

const noDiseaseProbability =
    document.getElementById("noDiseaseProbability");

const diseaseProgress =
    document.getElementById("diseaseProgress");

const noDiseaseProgress =
    document.getElementById("noDiseaseProgress");

const scoreProgress =
    document.getElementById("scoreProgress");


/* -----------------------------------------------------
   SEGMENTED CONTROLS
----------------------------------------------------- */

const segments =
    document.querySelectorAll(".segment");

segments.forEach(segment => {

    segment.addEventListener("click", () => {

        const field =
            segment.dataset.field;

        const value =
            segment.dataset.value;


        // Remove active state
        document
            .querySelectorAll(
                `.segment[data-field="${field}"]`
            )
            .forEach(button => {

                button.classList.remove("active");

            });


        // Activate selected button
        segment.classList.add("active");


        // Update hidden input
        document.getElementById(field).value =
            value;

    });

});


/* -----------------------------------------------------
   ANIMATE NUMBER
----------------------------------------------------- */

function animateNumber(
    element,
    target,
    duration = 1200
) {

    const start = performance.now();

    function update(currentTime) {

        const elapsed =
            currentTime - start;

        const progress =
            Math.min(elapsed / duration, 1);

        const eased =
            1 - Math.pow(1 - progress, 3);

        const current =
            target * eased;

        element.textContent =
            `${current.toFixed(1)}%`;

        if (progress < 1) {

            requestAnimationFrame(update);

        } else {

            element.textContent =
                `${target.toFixed(2)}%`;

        }

    }

    requestAnimationFrame(update);
}


/* -----------------------------------------------------
   RISK RING
----------------------------------------------------- */

function updateRiskRing(value) {

    const circumference = 515;

    const offset =
        circumference -
        (value / 100) * circumference;

    scoreProgress.style.strokeDashoffset =
        offset;

}


/* -----------------------------------------------------
   SHOW RESULT
----------------------------------------------------- */

function showResult(data) {

    emptyState.classList.add("hidden");

    resultContent.classList.remove("hidden");


    const risk =
        Number(data.heart_disease_probability);

    const noRisk =
        Number(data.no_heart_disease_probability);


    /* Risk score */

    animateNumber(
        riskPercentage,
        risk
    );


    updateRiskRing(risk);


    /* Probability values */

    animateNumber(
        diseaseProbability,
        risk
    );

    animateNumber(
        noDiseaseProbability,
        noRisk
    );


    /* Progress bars */

    setTimeout(() => {

        diseaseProgress.style.width =
            `${risk}%`;

        noDiseaseProgress.style.width =
            `${noRisk}%`;

    }, 150);


    /* Result state */

    if (data.prediction === 1) {

        riskBadge.textContent =
            "HIGHER RISK";

        riskBadge.classList.remove("low");

        riskTitle.textContent =
            "Higher Risk Detected";

        riskDescription.textContent =
            "The model identifies a higher probability " +
            "of heart disease based on the provided " +
            "parameters.";

    } else {

        riskBadge.textContent =
            "LOWER RISK";

        riskBadge.classList.add("low");

        riskTitle.textContent =
            "Lower Risk Detected";

        riskDescription.textContent =
            "The model identifies a lower probability " +
            "of heart disease based on the provided " +
            "parameters.";

    }

}


/* -----------------------------------------------------
   FORM SUBMISSION
----------------------------------------------------- */

form.addEventListener(
    "submit",
    async event => {

        event.preventDefault();


        /* Loading state */

        predictButton.classList.add("loading");

        predictButton.querySelector(
            ".button-text"
        ).textContent =
            "Analyzing...";


        /* Gather input */

        const patientData = {

            age:
                Number(
                    document.getElementById("age").value
                ),

            sex:
                Number(
                    document.getElementById("sex").value
                ),

            chest_pain_type:
                Number(
                    document.getElementById(
                        "chest_pain_type"
                    ).value
                ),

            resting_blood_pressure:
                Number(
                    document.getElementById(
                        "resting_blood_pressure"
                    ).value
                ),

            cholesterol:
                Number(
                    document.getElementById(
                        "cholesterol"
                    ).value
                ),

            fasting_blood_sugar:
                Number(
                    document.getElementById(
                        "fasting_blood_sugar"
                    ).value
                ),

            resting_ecg:
                Number(
                    document.getElementById(
                        "resting_ecg"
                    ).value
                ),

            max_heart_rate:
                Number(
                    document.getElementById(
                        "max_heart_rate"
                    ).value
                ),

            exercise_induced_angina:
                Number(
                    document.getElementById(
                        "exercise_induced_angina"
                    ).value
                ),

            st_depression:
                Number(
                    document.getElementById(
                        "st_depression"
                    ).value
                ),

            st_slope:
                Number(
                    document.getElementById(
                        "st_slope"
                    ).value
                ),

            num_major_vessels:
                Number(
                    document.getElementById(
                        "num_major_vessels"
                    ).value
                ),

            thalassemia:
                Number(
                    document.getElementById(
                        "thalassemia"
                    ).value
                )

        };


        try {

            /* API request */

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                patientData
                            )
                    }
                );


            if (!response.ok) {

                throw new Error(
                    "Prediction request failed."
                );

            }


            const data =
                await response.json();


            /* Show result */

            showResult(data);


        } catch (error) {

            console.error(error);

            alert(
                "Unable to connect to the prediction server. " +
                "Please make sure FastAPI is running."
            );

        } finally {

            predictButton.classList.remove(
                "loading"
            );

            predictButton.querySelector(
                ".button-text"
            ).textContent =
                "Analyze Heart Risk";

        }

    }
);
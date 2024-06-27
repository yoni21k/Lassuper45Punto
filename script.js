const results = [];
const ctx = document.getElementById('resultsChart').getContext('2d');
const analysisOutput = document.getElementById('analysis-output');
const predictionOutput = document.getElementById('prediction-output');
const resultsContainer = document.getElementById('resultsContainer');
const apiStatus = document.getElementById('apiStatus');

const resultsChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Resultados',
            data: [],
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false
        }]
    },
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

document.getElementById('add-result-button').addEventListener('click', () => {
    const resultInput = document.getElementById('result');
    const result = parseFloat(resultInput.value);

    if (!isNaN(result) && result > 0) {
        results.push(result);
        updateChart();
        addResultBlock(result);
        analyzeResults();
        if (results.length >= 10) {
            trainAndPredict();
        }
        resultInput.value = '';
    } else {
        alert('Por favor, ingrese un resultado válido.');
    }
});

async function fetchResultsFromAPI() {
    try {
        const response = await fetch('https://sentry.io/api/1313749/store/'); // Cambia esta URL a la correcta
        const data = await response.json();
        updateAPIStatus(true);
        return data.results; // Ajusta esto según la estructura de la respuesta de la API
    } catch (error) {
        console.error('Error fetching results from API:', error);
        updateAPIStatus(false);
        return [];
    }
}

function updateAPIStatus(isOnline) {
    if (isOnline) {
        apiStatus.textContent = 'API en línea';
        apiStatus.className = 'api-status online';
    } else {
        apiStatus.textContent = 'API fuera de servicio';
        apiStatus.className = 'api-status offline';
    }
}

async function initialize() {
    const apiResults = await fetchResultsFromAPI();
    if (apiResults.length > 0) {
        apiResults.forEach(result => {
            if (!isNaN(result) && result > 0) {
                results.push(result);
                updateChart();
                addResultBlock(result);
            }
        });
        if (results.length >= 10) {
            await trainAndPredict();
        }
    }
    await loadModel();
}

function updateChart() {
    resultsChart.data.labels.push(results.length);
    resultsChart.data.datasets[0].data.push(results[results.length - 1]);
    resultsChart.update();
}

function addResultBlock(result) {
    const block = document.createElement('div');
    block.className = 'result-block';
    block.textContent = result.toFixed(2);
    block.style.backgroundColor = getColor(result);
    resultsContainer.appendChild(block);
}

function getColor(value) {
    if (value < 2) {
        return '#00BFFF'; // Celeste para valores < 2
    } else if (value < 10) {
        return '#8A2BE2'; // Violeta para valores entre 2 y 9.99
    } else {
        return '#FF00FF'; // Fucsia para valores >= 10
    }
}

function analyzeResults() {
    if (results.length >= 5) {
        const recentResults = results.slice(-5);
        const average = recentResults.reduce((acc, val) => acc + val, 0) / recentResults.length;
        const advice = average > 2 ? 'Es un buen momento para apostar.' : 'Es mejor esperar.';
        analysisOutput.textContent = `Promedio de los últimos 5 resultados: ${average.toFixed(2)}x. ${advice}`;
    } else {
        analysisOutput.textContent = 'Ingrese al menos 5 resultados para ver el análisis.';
    }
}

async function trainAndPredict() {
    // Normalizar los datos
    const maxResult = Math.max(...results);
    const normalizedResults = results.map(r => r / maxResult);

    // Preparar los datos
    const xs = tf.tensor2d(normalizedResults.slice(0, -1), [results.length - 1, 1]);
    const ys = tf.tensor2d(normalizedResults.slice(1), [results.length - 1, 1]);

    // Crear el modelo
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    // Entrenar el modelo de manera optimizada
    const trainingStartTime = performance.now();
    await model.fit(xs, ys, {
        epochs: 100, // Reducir el número de épocas
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                const trainingEndTime = performance.now();
                const trainingTime = (trainingEndTime - trainingStartTime) / 1000;
                if (trainingTime > 5) {
                    model.stopTraining = true; // Detener el entrenamiento si tarda más de 5 segundos
                }
            }
        }
    });

    // Guardar el modelo en localStorage
    await model.save(tf.io.withSaveHandler(async (data) => {
        localStorage.setItem('model', JSON.stringify(data));
        return { success: true };
    }));

    // Hacer una predicción rápida
    const nextResult = await model.predict(tf.tensor2d([normalizedResults[normalizedResults.length - 1]], [1, 1]), { batch_size: 1 }).data();
    const denormalizedNextResult = nextResult[0] * maxResult;

    predictionOutput.textContent = `Predicción del próximo resultado: ${denormalizedNextResult.toFixed(2)}x.`;
}


async function loadModel() {
    const modelData = localStorage.getItem('model');
    if (modelData) {
        const model = await tf.loadLayersModel(tf.io.withLoadHandler(async () => {
            return JSON.parse(modelData);
        }));
        predictionOutput.textContent = 'Modelo cargado. Introduzca resultados para predecir.';
        return model;
    }
    predictionOutput.textContent = 'No hay modelo guardado. Introduzca resultados para entrenar.';
    return null;
}

// Cargar el modelo y resultados de la API si existen
window.onload = async () => {
    await initialize();
};

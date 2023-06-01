import { TRAINING_DATA } from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js'


const button = document.getElementById('button')
const prediction = document.getElementById('prediction')
const info = document.getElementById('info')
const canvas = document.getElementById('canvas').getContext('2d')


function logProgress(epoch, logs) {
  info.innerText += `\n E${epoch + 1} A ${Math.sqrt(logs.acc).toFixed(2)} V ${Math.sqrt(logs.val_acc).toFixed(2)}`
}


function drawImage(digit, model, inputs, outputs) {
  var imageData = canvas.getImageData(0, 0, 28, 28)
  for(let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255 // Red Channel
    imageData.data[i * 4 + 1] = digit[i] * 255 // Green Channel
    imageData.data[i * 4 + 2] = digit[i] * 255 // Blue Channel
    imageData.data[i * 4 + 3] = 255 // Alpha Channel
  }
  
  canvas.putImageData (imageData, 0, 0)
  setTimeout(() => evaluate(model, inputs, outputs), 2000)
}


function prepare(inputs, outputs) {
  const timeStart = performance.now()
  
  tf.util.shuffleCombo(inputs, outputs)

  const inputsTensor = tf.tensor2d(inputs)
  const outputsTensor = tf.oneHot(tf.tensor1d(outputs, 'int32'), 10)

  const model = tf.sequential()
  model.add(tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }))
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }))
  model.summary()
  
  const modelPrepareTime = performance.now() - timeStart
  info.innerText += `\n ⏳ ${Math.round(modelPrepareTime)}MS`
  
  return { model, inputsTensor, outputsTensor }
}


async function train(model, inputsTensor, outputsTensor) {
  const timeStart = performance.now()

  model.compile ({ 
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  const results = await model.fit(inputsTensor, outputsTensor, {
    callbacks: { onEpochEnd: logProgress },
    shuffle: true,           // Ensure data is shuffled in case it was in an order
    validationSplit: 0.2,
    batchSize: 512,          // As we have a lot of training data, batch size is set to 64
    epochs: 50               // Go over the data 10 times
  })
  
  inputsTensor.dispose()
  outputsTensor.dispose()
  
  const modelTrainTime = performance.now() - timeStart
  info.innerText += `\n\n ⏳ ${Math.round(modelTrainTime)}MS`
}


async function evaluate(model, inputs, outputs, first) {
  const timeStart = performance.now()
  
  const offset = Math.floor((Math.random() * inputs.length))
  
  const predictionTensor = tf.tidy(() => {
    const newInput = tf.tensor1d(inputs[offset]).expandDims()
    
    const output = model.predict(newInput)
    
    return output.squeeze().argMax()
  })
  
  const result = await predictionTensor.array()
  predictionTensor.dispose()
  
  prediction.innerText = result
  prediction.setAttribute('class', (result === outputs[offset]) ? 'correct' : 'wrong');
  drawImage(inputs[offset], model, inputs, outputs)
  
  const modelEvaluateTime = performance.now() - timeStart
  if (first) {
    info.innerText += `\n ⏳ ${Math.round(modelEvaluateTime)}MS`
  }
}


button.addEventListener('click', async event => {
  event.target.remove()

  info.innerText = '⎯⎯⎯⎯⎯⎯⎯ PREPARE ⎯⎯⎯⎯⎯⎯⎯\n'
  const { inputs, outputs } = TRAINING_DATA
  const { model, inputsTensor, outputsTensor } = prepare(inputs, outputs)
  
  info.innerText += '\n\n ⎯⎯⎯⎯⎯⎯⎯ TRAIN ⎯⎯⎯⎯⎯⎯⎯\n'
  await train(model, inputsTensor, outputsTensor)
  
  info.innerText += '\n\n ⎯⎯⎯⎯⎯⎯⎯ EVALUATE ⎯⎯⎯⎯⎯⎯⎯\n'
  await evaluate(model, inputs, outputs, true)
  
  info.innerText += '\n\n ⎯⎯⎯⎯⎯⎯⎯ MEMORY ⎯⎯⎯⎯⎯⎯⎯\n'
  info.innerText += `\n 💾 ${(tf.memory().numBytes / 1000000).toFixed(2)}MB (${tf.memory().numTensors})`
})
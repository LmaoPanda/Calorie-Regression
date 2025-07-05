import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

calories_dataset = pd.read_csv("Data/calories.csv")
exercise_dataset = pd.read_csv("Data/exercise.csv")
#print(calories_dataset)

training_dataset = pd.merge(calories_dataset, exercise_dataset, how='outer').drop(columns="User_ID")
#print(training_dataset.corr(numeric_only = True))
"""
Correlation with Calories to Duration, Heart_Rate Body_Temp 
"""
#sns.pairplot(training_dataset, x_vars=["Calories", "Duration", "Heart_Rate", "Body_Temp"], y_vars=["Calories", "Duration", "Heart_Rate", "Body_Temp"])
#plt.show()

def makePlots(df, featureNames, labelName, modelOutput, sampleSize = 500):
    randomSample = df.sample(sampleSize).copy()
    randomSample.reset_index()
    weights, bias, epochs, sqrMeanErr = modelOutput

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Correlation of Variables vs Calories", fontsize = 16)
    
    

    #plotData(randomSample, featureNames, labelName, fig)
    plotModels(randomSample, featureNames, weights, bias, fig, axes)
    plotLoss(epochs, sqrMeanErr, fig, axes)
    
    plt.show()

    return

def plotLoss(epochs, sqrMeanErr, fig, axes):
    axes[0].plot(epochs, sqrMeanErr, color = 'black', linewidth = 3)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Mean Squared Error")
    axes[0].set_ylim([sqrMeanErr.min() * 0.8, sqrMeanErr.max()*1.2])

    return

def plotModels(df, featuresNames, weights, bias, fig, axes):
    df['Calories Predicted'] = bias[0]

    for i, feature in enumerate(featuresNames):
        df['Calories Predicted'] += weights[i][0] * df[feature]

    #sns.pairplot(df, x_vars=["Calories Predicted"], y_vars=["Duration", "Heart_Rate", "Body_Temp"])
    xFeature = ["Duration", "Heart_Rate", "Body_Temp"]
    yFeature = "Calories Predicted"
    for i in range(len(featuresNames)):
        axes[i+1].scatter(df[xFeature[i]], df[yFeature])
        axes[i+1].set_xlabel(xFeature[i])
        axes[i+1].set_ylabel(yFeature)
        axes[i+1].set_title(f"{yFeature} vs. {xFeature[i]}")

    return

def modelInfo(featureNames, label, modelOutput):
    weights = modelOutput[0]
    bias = modelOutput[1]

    banner = ('-' * 80) + "\n" + "|" + "MODEL INFO".center(78) + "|" + "\n" + "-" * 80
    info = "" 
    equation = label + " = "

    for index, feature in enumerate(featureNames):
        info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
        equation = equation + "{:.3f} * {} + ". format(weights[index][0], feature)
    info = info + "Bias: {:.3f}\n".format(bias[0])
    equation = equation + "{:.3f}\n".format(bias[0])

    return banner + "\n" + info + "\n" + equation

def buildModel(learnRate, featureNum):
    inputs = keras.Input(shape=(featureNum,))
    outputs = keras.layers.Dense(units=1)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learnRate),
                    loss="mean_squared_error",
                    metrics=[keras.metrics.RootMeanSquaredError()])

    return model

def trainModel(model, features, label, epochs, batchSize):
    history = model.fit(
        x = features,
        y = label,
        batch_size = batchSize,
        epochs=epochs
    )
    trainedWeight = model.get_weights()[0]
    trainedBias = model.get_weights()[1]

    epochs = history.epoch

    totalHistory = pd.DataFrame(history.history)

    rmse = totalHistory["root_mean_squared_error"]

    return trainedWeight, trainedBias, epochs, rmse

def runModel(df, featureNames, labelName, learnRate, epochs, batchSize):
    print('INFO: starting training experiment with features={} and label={}\n'.format(featureNames, labelName))
    featureNum = len(featureNames)

    features = df.loc[:, featureNames].values
    label = df[labelName].values

    model = buildModel(learnRate, featureNum)
    modelOutput = trainModel(model, features, label, epochs, batchSize)

    print('\nSUCCESS: training exp0eriment complete\n')
    print('{}'.format(modelInfo(featureNames, labelName, modelOutput)))
    makePlots(df, featureNames, labelName, modelOutput)

    return model

learningRate = 0.0025
epochs = 50
batchSize = 25

# Specify the feature and the label.
features = ['Duration', 'Heart_Rate', 'Body_Temp' ]
label = 'Calories'

model_1 = runModel(training_dataset, features, label, learningRate, epochs, batchSize)

def buildBatch(df, batchSize):
    batch = df.sample(n=batchSize).copy()
    batch.set_index(np.arange(batchSize), inplace=True)
    return batch

def predictCaloriesBurnt(model, df, features, label, batchSize = 50):
    batch = buildBatch(df, batchSize)
    predictedValues = model.predict_on_batch(x=batch.loc[:,features].values)
    data = {"Calories Predicted": [], "Calories":[], "L1 Loss": [], 
             features[0]: [], features[1]: [], features[2]: []}
    for i in range(batchSize):
        predicted = predictedValues[i][0]
        observed = batch.at[i, label]
        data["Calories Predicted"].append((predicted))
        data["Calories"].append((observed))
        data["L1 Loss"].append((abs(observed - predicted)))
        data[features[0]].append(batch.at[i, features[0]])
        data[features[1]].append(batch.at[i, features[1]])
        data[features[2]].append(batch.at[i, features[2]])

    output_df = pd.DataFrame(data)
    return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

output = predictCaloriesBurnt(model_1, training_dataset, features, label)
show_predictions(output)
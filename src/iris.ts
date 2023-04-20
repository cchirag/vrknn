import { KNN, VRKNN, Dataset } from "./lib";
import iris_dataset from "./data/iris.json";
console.log("Iris Dataset Predictions");

interface IrisDataPoint {
  sepalLength: number;
  sepalWidth: number;
  petalLength: number;
  petalWidth: number;
  species: string;
}

const iris_data: IrisDataPoint[] = iris_dataset;

const [training_data, testing_data, testing_labels] = Dataset.train_test_split({
  data: iris_data,
  training_entities: ["sepalLength", "sepalWidth"],
  testing_entity: "species",
  test_size: 0.2,
  random_state: 42,
});
const iris_k_value = 3;
const radius_delta = 0.1;

const knn = new KNN({
  k: iris_k_value,
});

knn.fit({
  training_data: training_data,
});

const predictions = knn.predict({
  testing_data: testing_data,
});

const accuracy_knn = knn.evaluate({
  predictions: predictions,
  testing_labels: testing_labels,
});

const vrknn = new VRKNN({
  k: iris_k_value,
  radius_delta: radius_delta,
});

vrknn.fit({
  training_data: training_data,
});

const vr_predictions = vrknn.predict({
  testing_data: testing_data,
});

const accuracy_vrknn = vrknn.evaluate({
  predictions: vr_predictions,
  testing_labels: testing_labels,
});

console.log("Total Iris Dataset: ", iris_data.length);
console.log("Length of training data of Iris Dataset: ", training_data.length);
console.log("Length of testing data of Iris Dataset: ", testing_data.length);
console.log("K value of Iris Dataset: ", iris_k_value);
console.log("Radius delta of Iris Dataset: ", radius_delta);
console.log("Accuracy of KNN: ", accuracy_knn);
console.log(
  "Time taken to predict using KNN: ",
  knn.time_taken_to_predict,
  "ms"
);
console.log("Accuracy of VRKNN: ", accuracy_vrknn);
console.log(
  "Time taken to predict using VRKNN: ",
  vrknn.time_taken_to_predict,
  "ms"
);

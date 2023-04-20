import magic_gamma_dataset from "./data/magic_gamma.json";
import { KNN, VRKNN, Dataset } from "./lib";

// MAGIC Gamma Telescope Dataset
interface MagicGammaTelescopeDataPoint {
  fLength: number;
  fWidth: number;
  class: string;
}

const magic_gamma_telescope_data: MagicGammaTelescopeDataPoint[] =
  magic_gamma_dataset as MagicGammaTelescopeDataPoint[];

const [
  training_data_magic_gamma_telescope,
  testing_data_magic_gamma_telescope,
  testing_labels_magic_gamma_telescope,
] = Dataset.train_test_split({
  data: magic_gamma_telescope_data,
  training_entities: ["fLength", "fWidth"],
  testing_entity: "class",
  test_size: 0.1,
  random_state: 42,
});

const magic_gamma_telescope_k_value = 10;
const magic_gamma_telescope_radius_delta = 0.1;

const knn_magic_gamma_telescope = new KNN({
  k: magic_gamma_telescope_k_value,
});

knn_magic_gamma_telescope.fit({
  training_data: training_data_magic_gamma_telescope,
});

const predictions_magic_gamma_telescope = knn_magic_gamma_telescope.predict({
  testing_data: testing_data_magic_gamma_telescope,
});

const accuracy_knn_magic_gamma_telescope = knn_magic_gamma_telescope.evaluate({
  predictions: predictions_magic_gamma_telescope,
  testing_labels: testing_labels_magic_gamma_telescope,
});

const vrknn_magic_gamma_telescope = new VRKNN({
  k: magic_gamma_telescope_k_value,
  radius_delta: magic_gamma_telescope_radius_delta,
});

vrknn_magic_gamma_telescope.fit({
  training_data: training_data_magic_gamma_telescope,
});

const vr_predictions_magic_gamma_telescope =
  vrknn_magic_gamma_telescope.predict({
    testing_data: testing_data_magic_gamma_telescope,
  });

const accuracy_vrknn_magic_gamma_telescope =
  vrknn_magic_gamma_telescope.evaluate({
    predictions: vr_predictions_magic_gamma_telescope,
    testing_labels: testing_labels_magic_gamma_telescope,
  });

console.log("\nMAGIC Gamma Telescope Dataset Predictions");
console.log(
  "Total MAGIC Gamma Telescope Dataset: ",
  magic_gamma_telescope_data.length
);
console.log(
  "Length of training data of MAGIC Gamma Telescope Dataset: ",
  training_data_magic_gamma_telescope.length
);

console.log(
  "Length of testing data of MAGIC Gamma Telescope Dataset: ",
  testing_data_magic_gamma_telescope.length
);

console.log(
  "K value of MAGIC Gamma Telescope Dataset: ",
  magic_gamma_telescope_k_value
);

console.log(
  "Radius delta of MAGIC Gamma Telescope Dataset: ",
  magic_gamma_telescope_radius_delta
);

console.log("Accuracy of KNN: ", accuracy_knn_magic_gamma_telescope);
console.log(
  "Time taken to predict using KNN: ",
  knn_magic_gamma_telescope.time_taken_to_predict,
  "ms"
);

console.log("Accuracy of VRKNN: ", accuracy_vrknn_magic_gamma_telescope);

console.log(
  "Time taken to predict using VRKNN: ",
  vrknn_magic_gamma_telescope.time_taken_to_predict,
  "ms"
);

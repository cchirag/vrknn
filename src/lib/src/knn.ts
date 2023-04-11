import { TestDatapoint, TrainingDataPoint } from "../../models";
import { best_of_all, distance } from "../../utils";

export class KNN {
  private training_data: TrainingDataPoint[] = [];
  private k: number;
  public time_taken_to_predict: number = 0;

  constructor(params: { k: number }) {
    this.k = params.k;
  }

  public fit(params: { training_data: TrainingDataPoint[] }): void {
    const { training_data } = params;
    this.training_data = training_data;

    if (this.training_data.length === 0)
      throw new Error("Training data must not be empty");
    if (this.k > this.training_data.length)
      throw new Error("K must be less than the number of training data");
  }

  public predict(params: { testing_data: TestDatapoint[] }): string[] {
    const start_time = Date.now();
    const { testing_data } = params;
    const predictions: string[] = [];
    testing_data.forEach((test_data_point) => {
      const distances: [string, number][] = [];
      this.training_data.forEach((training_data_point) => {
        distances.push([
          training_data_point.label,
          distance({
            point1: test_data_point.coordinates,
            point2: training_data_point.coordinates,
          }),
        ]);
      });
      const sorted_distances = distances.sort((a, b) => a[1] - b[1]);
      const k_nearest_neighbors = sorted_distances.slice(0, this.k);
      const labels = k_nearest_neighbors.map((neighbor) => neighbor[0]);
      predictions.push(best_of_all({ labels }));
    });
    this.time_taken_to_predict = Date.now() - start_time;
    return predictions;
  }

  public evaluate(params: {
    predictions: string[];
    testing_labels: string[];
  }): number {
    const { predictions, testing_labels } = params;

    let correct_predictions = 0;
    predictions.forEach((prediction, index) => {
      if (prediction === testing_labels[index]) correct_predictions++;
    });
    return (correct_predictions / predictions.length) * 100;
  }
}

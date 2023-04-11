import { best_of_all, distance } from "../../utils";
import { TestDatapoint, TrainingDataPoint } from "../../models";

interface Bucket {
  id: number;
  data: TrainingDataPoint[];
}
class HashTable {
  private data: TrainingDataPoint[] = [];
  private buckets: Bucket[] = [];

  private hash_function(params: { data_point: TrainingDataPoint }): number {
    const { data_point } = params;
    const { coordinates } = data_point;
    const hash = Math.floor(coordinates[0]);
    return hash;
  }

  public insert(params: { data_point: TrainingDataPoint }): void {
    const { data_point } = params;
    const hash = this.hash_function({ data_point });
    const bucket = this.buckets.find((bucket) => bucket.id === hash);
    // Sort the data points in the bucket by their y coordinate
    if (bucket) {
      bucket.data.push(data_point);
      bucket.data.sort((a, b) => a.coordinates[1] - b.coordinates[1]);
      return;
    }
    // Create a new bucket if it doesn't exist
    const new_bucket: Bucket = {
      id: hash,
      data: [data_point],
    };
    this.buckets.push(new_bucket);
  }

  public data_points_in_range(params: {
    max_x: number;
    min_x: number;
    min_y: number;
    max_y: number;
  }): TrainingDataPoint[] {
    const { max_x, min_x, max_y, min_y } = params;
    const data_points_in_range: TrainingDataPoint[] = [];
    this.buckets.forEach((bucket) => {
      if (bucket.id >= min_x && bucket.id <= max_x) {
        bucket.data.forEach((data_point) => {
          if (
            data_point.coordinates[1] >= min_y &&
            data_point.coordinates[1] <= max_y
          ) {
            data_points_in_range.push(data_point);
          }
        });
      }
    });
    return data_points_in_range;
  }
}

export class VRKNN {
  private hash_table: HashTable = new HashTable();
  private training_data: TrainingDataPoint[] = [];
  private k: number;
  private radius_delta: number;
  public time_taken_to_predict: number = 0;

  constructor(params: { k: number; radius_delta: number }) {
    this.k = params.k;
    this.radius_delta = params.radius_delta;
  }

  public fit(params: { training_data: TrainingDataPoint[] }): void {
    const { training_data } = params;
    this.training_data = training_data;

    if (this.training_data.length === 0)
      throw new Error("Training data must not be empty");

    if (this.k > this.training_data.length)
      throw new Error("K must be less than the number of training data");

    this.training_data.forEach((data_point) => {
      if (data_point.coordinates.length !== 2)
        throw new Error("Data points must have 2 coordinates");
    });

    this.training_data.forEach((data_point) => {
      this.hash_table.insert({ data_point });
    });
  } // TODO

  public predict(params: { testing_data: TestDatapoint[] }): string[] {
    const { testing_data } = params;

    const predictions: string[] = [];

    if (this.training_data.length === 0)
      throw new Error("Must fit the model before predicting");

    const start_time = Date.now();

    testing_data.forEach((test_datapoint) => {
      const nearest_neighbors: TrainingDataPoint[][] = [];
      const distances: [string, number][] = [];
      while (nearest_neighbors.length < this.k) {
        testing_data.forEach((test_datapoint) => {
          var x_min = test_datapoint.coordinates[0] - this.radius_delta;
          var x_max = test_datapoint.coordinates[0] + this.radius_delta;
          var y_min = test_datapoint.coordinates[1] - this.radius_delta;
          var y_max = test_datapoint.coordinates[1] + this.radius_delta;
          var data_points_in_range = this.hash_table.data_points_in_range({
            max_x: x_max,
            min_x: x_min,
            max_y: y_max,
            min_y: y_min,
          });
          nearest_neighbors.push(data_points_in_range);
        });
      }
      nearest_neighbors.forEach((neighbors) => {
        neighbors.forEach((neighbor) => {
          distances.push([
            neighbor.label,
            distance({
              point1: [neighbor.coordinates[0], neighbor.coordinates[1]],
              point2: [
                test_datapoint.coordinates[0],
                test_datapoint.coordinates[1],
              ],
            }),
          ]);
        });
      });

      distances.sort((a, b) => a[1] - b[1]);
      const k_nearest_neighbors = distances.slice(0, this.k);
      const labels = k_nearest_neighbors.map((neighbor) => neighbor[0]);
      predictions.push(best_of_all({ labels }));
    });

    this.time_taken_to_predict = Date.now() - start_time;
    return predictions;
  } // TODO

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
  } // TODO
}
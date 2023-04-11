export const distance = (params: {
  point1: number[];
  point2: number[];
}): number => {
  const { point1, point2 } = params;
  if (point1.length !== point2.length)
    throw new Error("Points must be of the same dimension");
  let sum = 0;
  for (let i = 0; i < point1.length; i++) {
    sum += Math.pow(point1[i] - point2[i], 2);
  }
  return Math.sqrt(sum);
};

export const best_of_all = (params: { labels: string[] }): string => {
  const { labels } = params;
  const label_counts: { [key: string]: number } = {};
  labels.forEach((label) => {
    if (label_counts[label]) label_counts[label]++;
    else label_counts[label] = 1;
  });
  const sorted_labels = Object.entries(label_counts).sort(
    (a, b) => b[1] - a[1]
  );
  return sorted_labels[0][0];
};

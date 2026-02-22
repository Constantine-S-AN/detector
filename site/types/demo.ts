export type DemoIndexItem = {
  sample_id: string;
  label: "faithful" | "hallucinated";
  prompt_preview: string;
  groundedness_score: number;
  predicted_label: number;
  abstain_flag: boolean;
  detail_path: string;
};

export type DemoDetail = {
  sample_id: string;
  label: "faithful" | "hallucinated";
  prompt: string;
  answer: string;
  prediction: {
    groundedness_score: number;
    predicted_label: number;
    confidence: number;
    abstain_flag: boolean;
  };
  features: Record<string, number | boolean | string>;
  top_influential: Array<{
    train_id: string;
    score: number;
    text: string;
    meta: Record<string, string | number | boolean>;
  }>;
};

import { PageIntro } from "@/components/page-intro";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const quickCommands = [
  "make setup",
  "make lint",
  "make test",
  "make demo",
  "make build-site",
  "make serve-api",
];

const pipelineStages = [
  "scripts/build_controlled_dataset.py",
  "scripts/run_attribution.py",
  "scripts/build_features.py",
  "scripts/train_detector.py",
  "scripts/evaluate_detector.py",
  "scripts/export_demo_assets.py",
  "scripts/write_run_manifest.py",
];

export default function DocsPage() {
  return (
    <main className="space-y-6">
      <PageIntro
        eyebrow="Runbook"
        title="Docs"
        description="Operational notes for local development, reproducible evaluation, and static deployment to GitHub Pages."
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Commands</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-2 rounded-xl bg-slate-900 p-4 text-sm text-slate-100">
              {quickCommands.map((command) => (
                <code key={command}>{command}</code>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Execution Pipeline</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="grid gap-2 text-sm text-slate-700">
              {pipelineStages.map((stage) => (
                <li key={stage} className="rounded-lg bg-slate-100 px-3 py-2">
                  <code>{stage}</code>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>FULL Mode</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm leading-relaxed text-slate-700">
            <p>
              Start local API with <code>make serve-api</code>.
            </p>
            <p>
              Set frontend env var:
              <code> NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000</code>
            </p>
            <p>
              Open <code>/scan</code> and choose <code>FULL</code> mode.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>STATIC Mode</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm leading-relaxed text-slate-700">
            <p>
              Generate deterministic demo assets with <code>make demo</code>.
            </p>
            <p>
              The site reads precomputed JSON under <code>site/public/demo</code>.
            </p>
            <p>
              Build and export static pages using <code>make build-site</code>.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Artifacts</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2 text-sm text-slate-700 md:grid-cols-2">
          <p>
            <code>artifacts/metrics.json</code> for evaluation metrics and curve
            points.
          </p>
          <p>
            <code>artifacts/run_manifest.json</code> for reproducibility metadata.
          </p>
          <p>
            <code>site/public/demo/examples/*.json</code> for per-sample detail
            payloads.
          </p>
          <p>
            <code>site/public/demo/analysis.json</code> for analysis dashboard
            inputs.
          </p>
        </CardContent>
      </Card>
    </main>
  );
}

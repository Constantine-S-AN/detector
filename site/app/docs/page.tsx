import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const commands = [
  "make setup",
  "make test",
  "make demo",
  "make build-site",
  "make serve-api",
];

export default function DocsPage() {
  return (
    <main className="space-y-6">
      <section className="animate-rise rounded-2xl border border-[var(--ads-border)] bg-white/88 p-6 shadow-soft">
        <h1 className="text-3xl font-bold">Docs</h1>
        <p className="mt-2 text-sm text-slate-600">
          Run ADS locally or publish static assets via GitHub Pages.
        </p>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Commands</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 rounded-xl bg-slate-900 p-4 text-sm text-slate-100">
            {commands.map((command) => (
              <code key={command}>{command}</code>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Modes</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-slate-700">
          <p>
            <strong>FULL:</strong> Start local API with{" "}
            <code>make serve-api</code>, set
            <code> NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000</code>, then use
            Scan page.
          </p>
          <p>
            <strong>STATIC:</strong> Build and export demo assets; the site
            reads JSON under
            <code> site/public/demo</code> without any backend.
          </p>
        </CardContent>
      </Card>
    </main>
  );
}

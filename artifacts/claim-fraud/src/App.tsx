import { type FormEvent, type ReactNode, useMemo, useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import {
  Activity, ArrowRight, BarChart3, Check,
  ChevronDown, CircleAlert, ClipboardCheck, Database, Gauge, Info, LoaderCircle,
  RefreshCw, Shield, ShieldAlert, SlidersHorizontal, TriangleAlert, X,
} from 'lucide-react';
import {
  FraudFeature, FraudFeatureCatalog, FraudMetrics, FraudOverview, FraudPrediction,
  useCreateFraudPrediction, useGetFraudFeatureCatalog, useGetFraudMetrics,
  useGetFraudOverview, useHealthCheck,
} from '@workspace/api-client-react';
import { Link, Route, Switch, useLocation, Router as WouterRouter } from 'wouter';
import './App.css';

const queryClient = new QueryClient();

function numberFormat(value?: number) {
  if (value == null || !Number.isFinite(value)) return '—';
  return new Intl.NumberFormat('en-US').format(value);
}
function percent(value?: number, digits = 1) {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}
function titleCase(value: string) {
  return value.replace(/([A-Z])/g, ' $1').replace(/[_-]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
function InlineError({ retry }: { retry: () => void }) {
  return <div className="inline-error" role="alert"><CircleAlert size={17} /><span>We couldn’t load this information.</span><button onClick={retry} data-testid="button-retry"><RefreshCw size={14} /> Retry</button></div>;
}
function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-label="Loading" />;
}

function Shell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const screening = location === '/screening';
  const { data: health } = useHealthCheck();
  return <div className="app-frame">
    <aside className="sidebar">
      <Link href="/" className="brand" data-testid="link-home">
        <span className="brand-mark"><Shield size={19} strokeWidth={2.4} /></span>
        <span><b>ClaimSignal</b><small>INVESTIGATOR WORKSPACE</small></span>
      </Link>
      <div className="side-section-label">WORKSPACE</div>
      <nav className="side-nav" aria-label="Main navigation">
        <Link href="/" className={`side-link ${!screening ? 'active' : ''}`} data-testid="link-overview"><BarChart3 size={18} /><span>Model overview</span>{!screening && <span className="nav-pip" />}</Link>
        <Link href="/screening" className={`side-link ${screening ? 'active' : ''}`} data-testid="link-screening"><ClipboardCheck size={18} /><span>Screen a claim</span>{screening && <span className="nav-pip" />}</Link>
      </nav>
      <div className="sidebar-bottom">
        <div className="side-note-icon"><Info size={16} /></div>
        <p>Decision support, not a decision.</p>
        <small>Every alert needs a human review.</small>
        <div className="service-health"><span className={`health-dot ${health?.status === 'ok' || health?.status === 'healthy' ? 'good' : ''}`} /><span>Scoring service</span><span className="health-status">{health?.status ? titleCase(health.status) : 'Checking'}</span></div>
      </div>
      <div className="sidebar-foot">CLAIMS OPERATIONS <span>•</span> v1.0</div>
    </aside>
    <main className="main-shell">
      <header className="topbar">
        <div className="breadcrumbs"><span>Claim operations</span><span className="crumb-slash">/</span><strong>{screening ? 'Claim screening' : 'Model overview'}</strong></div>
        <div className="topbar-right"><span className="secure-label"><span className="secure-dot" /> INTERNAL REVIEW</span><span className="topbar-divider" /><span className="reviewer-chip">IR</span><span className="reviewer-label">Investigator</span></div>
      </header>
      <div className="page-content">{children}</div>
    </main>
  </div>;
}

function MetricCard({ label, value, foot, icon: Icon, tint }: { label: string; value: string; foot: string; icon: typeof Database; tint?: string }) {
  return <article className="metric-card">
    <div className="metric-card-top"><span>{label}</span><span className={`metric-icon ${tint || ''}`}><Icon size={17} /></span></div>
    <strong className="metric-value font-display">{value}</strong><span className="metric-foot">{foot}</span>
  </article>;
}

function ModelComparison({ metrics, overview }: { metrics?: FraudMetrics; overview?: FraudOverview }) {
  const [selected, setSelected] = useState<string | null>(null);
  if (!metrics) return <div className="comparison-empty"><BarChart3 size={24} /><p>Model comparison is unavailable.</p><small>Metrics will appear when evaluation data is available.</small></div>;
  const models = metrics.models || [];
  const activeModel = selected || metrics.selectedModel || overview?.selectedModel;
  const selectedMetric = models.find((m) => m.model === activeModel);
  return <div className="comparison-layout">
    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead><tr><th>MODEL</th><th>PRECISION</th><th>RECALL</th><th>F1</th><th>ROC AUC</th><th>PR AUC</th></tr></thead>
        <tbody>{models.map((m) => {
          const chosen = m.model === activeModel;
          return <tr key={m.model} className={chosen ? 'chosen-row' : ''} onClick={() => setSelected(m.model)} data-testid={`row-model-${m.model}`}>
            <td><button className="model-select" onClick={() => setSelected(m.model)} data-testid={`button-model-${m.model}`}><span className={`radio-dot ${chosen ? 'checked' : ''}`} />{m.model}{m.model === (metrics.selectedModel || overview?.selectedModel) && <span className="selected-badge">SELECTED</span>}</button></td>
            <td>{percent(m.precision)}</td><td>{percent(m.recall)}</td><td className="metric-strong">{percent(m.f1)}</td><td>{percent(m.rocAuc)}</td><td>{percent(m.prAuc)}</td>
          </tr>;
        })}</tbody>
      </table>
      {models.length === 0 && <div className="table-empty"><Info size={16} /> No model results were returned.</div>}
      <p className="table-footnote"><span className="legend-dot" /> Selection is based on the strategy shown above. Metrics are from the held-out evaluation set.</p>
    </div>
    {selectedMetric && <aside className="model-detail">
      <div className="model-detail-head"><span className="eyebrow">SELECTED MODEL</span><ChevronDown size={15} /></div>
      <h3>{selectedMetric.model}</h3><p className="detail-copy">Threshold <b>{Number.isFinite(selectedMetric.threshold) ? selectedMetric.threshold.toFixed(3) : '—'}</b> · test set</p>
      <div className="cm-title">CONFUSION MATRIX <Info size={13} /></div>
      <div className="confusion-grid"><div className="cm-head" /><div className="cm-head">Predicted clear</div><div className="cm-head">Predicted flag</div>
        <div className="cm-label">Actual clear</div><div className="cm-cell">{numberFormat(selectedMetric.confusionMatrix?.trueNegative)}<small>True negative</small></div><div className="cm-cell caution">{numberFormat(selectedMetric.confusionMatrix?.falsePositive)}<small>False positive</small></div>
        <div className="cm-label">Actual fraud</div><div className="cm-cell caution">{numberFormat(selectedMetric.confusionMatrix?.falseNegative)}<small>False negative</small></div><div className="cm-cell positive">{numberFormat(selectedMetric.confusionMatrix?.truePositive)}<small>True positive</small></div>
      </div>
      <p className="cm-note">Counts reflect model outputs at the evaluation threshold—not confirmed claim outcomes.</p>
    </aside>}
  </div>;
}

function OverviewPage() {
  const overview = useGetFraudOverview();
  const metrics = useGetFraudMetrics();
  const catalog = useGetFraudFeatureCatalog();
  const data = overview.data;
  const metricData = metrics.data;
  return <div className="overview-page rise-in">
    <div className="page-heading">
      <div><div className="eyebrow"><span className="eyebrow-line" />MODEL &amp; DATASET</div><h1 className="font-display">A clearer view of<br className="mobile-break" /> claim risk.</h1><p className="page-subtitle">Evaluation context for investigator-led claim review.</p></div>
      <Link href="/screening" className="primary-button" data-testid="button-start-screening"><ClipboardCheck size={16} /> Screen a claim <ArrowRight size={15} /></Link>
    </div>
    {(overview.isLoading || catalog.isLoading) && <div className="stat-grid">{[1, 2, 3, 4].map((n) => <Skeleton key={n} className="stat-skeleton" />)}</div>}
    {overview.isError && <InlineError retry={() => overview.refetch()} />}
    {data && <section className="stat-grid" aria-label="Dataset summary">
      <MetricCard label="Claims evaluated" value={numberFormat(data.datasetRows)} foot="Rows in source dataset" icon={Database} />
      <MetricCard label="Fraud cases" value={numberFormat(data.fraudCases)} foot="Labeled in source dataset" icon={ShieldAlert} tint="warm" />
      <MetricCard label="Observed fraud rate" value={percent(data.fraudRate)} foot="Dataset prevalence" icon={Activity} tint="blue" />
      <MetricCard label="Available signals" value={numberFormat(data.featureCount)} foot={catalog.data ? `${catalog.data.featureCount} fields available for entry` : 'Model input features'} icon={SlidersHorizontal} tint="sand" />
    </section>}
    {catalog.isError && <div className="subtle-error"><CircleAlert size={14} /> Feature catalog unavailable; the field count above is from the model overview.</div>}
    <section className="section-panel model-panel">
      <div className="section-heading">
        <div><div className="eyebrow">EVALUATION</div><h2 className="font-display">Model comparison</h2><p>Compare held-out performance before reviewing a claim.</p></div>
        <div className="evaluation-meta"><span><span className="meta-label">SELECTED</span>{data?.selectedModel || metricData?.selectedModel || '—'}</span><span><span className="meta-label">TEST RECORDS</span>{metricData ? numberFormat(metricData.testRecords) : '—'}</span></div>
      </div>
      {metrics.isLoading && <div className="comparison-loading"><Skeleton /><Skeleton /><Skeleton /></div>}
      {metrics.isError && <InlineError retry={() => metrics.refetch()} />}
      {!metrics.isLoading && !metrics.isError && <ModelComparison metrics={metricData} overview={data} />}
    </section>
    {data && <section className="context-grid">
      <article className="context-panel">
        <div className="eyebrow">DECISION CONTEXT</div><div className="threshold-row"><span className="threshold-symbol"><Gauge size={20} /></span><div><span className="context-label">Decision threshold</span><strong className="font-mono">{Number.isFinite(data.decisionThreshold) ? data.decisionThreshold.toFixed(3) : '—'}</strong></div><span className="threshold-percent">{percent(data.decisionThreshold)}</span></div>
        <div className="context-rule" /><div className="context-label">Selection strategy</div><p className="strategy-text">{data.selectionStrategy || 'Not specified by the evaluation.'}</p>
      </article>
      <article className="caveat-panel"><div className="caveat-title"><TriangleAlert size={17} /><span>Read before interpreting</span></div>
        {data.caveats?.length ? <ul>{data.caveats.map((caveat, index) => <li key={`${index}-${caveat}`}>{caveat}</li>)}</ul> : <p>No caveats were provided with this overview.</p>}
        <div className="human-note"><Shield size={15} /><span>Scores support triage. They are not findings of fraud.</span></div>
      </article>
    </section>}
    <div className="source-note"><Info size={14} /><span>All figures reflect the currently supplied evaluation data. Dataset characteristics and limitations may affect how well results transfer to a specific claim.</span></div>
  </div>;
}

function initialValue(feature: FraudFeature): string {
  return '';
}
function FeatureField({ feature, value, onChange }: { feature: FraudFeature; value: string; onChange: (value: string) => void }) {
  const id = `feature-${feature.name}`;
  return <div className="field">
    <label htmlFor={id}>{feature.label}<span className="field-technical">{feature.name}</span></label>
    {feature.kind === 'categorical' ? <div className="select-wrap"><select id={id} value={value} onChange={(e) => onChange(e.target.value)} data-testid={`input-${feature.name}`}><option value="" disabled>Select {feature.label.toLowerCase()}</option>{(feature.options || []).map((option) => <option key={option} value={option}>{titleCase(option)}</option>)}</select><ChevronDown size={15} /></div> :
    <input id={id} type="number" value={value} min={feature.minimum ?? undefined} max={feature.maximum ?? undefined} step={feature.step ?? 'any'} placeholder={feature.minimum != null ? `Minimum ${feature.minimum}` : 'Enter a value'} onChange={(e) => onChange(e.target.value)} data-testid={`input-${feature.name}`} />}
    {feature.kind === 'numeric' && (feature.minimum != null || feature.maximum != null) && <small className="range-note">{feature.minimum != null && feature.maximum != null ? `Accepted range ${feature.minimum} to ${feature.maximum}` : feature.minimum != null ? `Minimum ${feature.minimum}` : `Maximum ${feature.maximum}`}</small>}
  </div>;
}

function PredictionResult({ result, onDismiss }: { result: FraudPrediction; onDismiss: () => void }) {
  const risk = String(result.riskLevel || '').toLowerCase();
  const riskLabel = risk ? titleCase(risk) : 'Unspecified';
  const riskScore = Number(result.fraudRiskScore);
  const circumference = 2 * Math.PI * 44;
  const dash = Number.isFinite(riskScore) ? Math.max(0, Math.min(1, riskScore)) * circumference : 0;
  return <section className={`result-panel risk-${risk || 'unknown'} rise-in`} aria-live="polite" data-testid="status-prediction-result">
    <div className="result-top"><div><span className="eyebrow">TRIAGE GUIDANCE</span><h2 className="font-display">Review signal</h2></div><button className="icon-button" aria-label="Dismiss result" onClick={onDismiss} data-testid="button-dismiss-result"><X size={17} /></button></div>
    <div className="result-score-row">
      <div className="score-visual"><svg viewBox="0 0 112 112" role="img" aria-label={`Fraud risk score ${percent(riskScore)}`}><circle cx="56" cy="56" r="44" className="score-track" /><circle cx="56" cy="56" r="44" className="score-arc" strokeDasharray={`${dash} ${circumference}`} transform="rotate(-90 56 56)" /></svg><div className="score-value"><b>{percent(riskScore)}</b><small>risk score</small></div></div>
      <div className="risk-summary"><span className="risk-label"><span className="risk-indicator" />{riskLabel} review signal</span><p>{result.recommendation || 'No recommendation was supplied. Use your standard review process.'}</p><div className="result-threshold">Decision threshold <b className="font-mono">{Number.isFinite(result.decisionThreshold) ? result.decisionThreshold.toFixed(3) : '—'}</b></div></div>
    </div>
    <div className="result-meta"><span>MODEL</span><strong>{result.model || 'Not specified'}</strong><span className="result-meta-spacer" /><span>INTERPRETATION</span><strong>Human review required</strong></div>
    {result.limitations?.length ? <div className="limitations"><div><Info size={15} /> Limitations</div><ul>{result.limitations.map((text, i) => <li key={`${i}-${text}`}>{text}</li>)}</ul></div> : <p className="limitations-quiet"><Info size={14} /> No additional limitations were returned. A score is not a finding of fraud.</p>}
    <div className="result-disclaimer"><Shield size={15} /> This score is a triage aid—not a fraud determination. Review the claim evidence independently.</div>
  </section>;
}

function ScreeningPage() {
  const catalog = useGetFraudFeatureCatalog();
  const overview = useGetFraudOverview();
  const prediction = useCreateFraudPrediction();
  const features = catalog.data?.features || [];
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<FraudPrediction | null>(null);
  const [validation, setValidation] = useState('');
  const grouped = useMemo(() => features.reduce<Record<string, FraudFeature[]>>((acc, feature) => {
    (acc[feature.group || 'Claim details'] ||= []).push(feature);
    return acc;
  }, {}), [features]);

  function setField(name: string, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
    if (validation) setValidation('');
  }
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const payload: Record<string, string | number> = {};
    for (const feature of features) {
      const raw = values[feature.name] ?? initialValue(feature);
      if (raw === '') { setValidation(`Enter a value for ${feature.label}.`); document.getElementById(`feature-${feature.name}`)?.focus(); return; }
      if (feature.kind === 'numeric') {
        const numeric = Number(raw);
        if (!Number.isFinite(numeric) || (feature.minimum != null && numeric < feature.minimum) || (feature.maximum != null && numeric > feature.maximum)) {
          setValidation(`Enter a valid value for ${feature.label}${feature.minimum != null && feature.maximum != null ? ` between ${feature.minimum} and ${feature.maximum}` : ''}.`);
          document.getElementById(`feature-${feature.name}`)?.focus(); return;
        }
        payload[feature.name] = numeric;
      } else {
        if (!feature.options?.includes(raw)) { setValidation(`Choose a supported value for ${feature.label}.`); return; }
        payload[feature.name] = raw;
      }
    }
    if (!features.length) { setValidation('No supported claim fields are available. Try again when the feature catalog is available.'); return; }
    setValidation('');
    setResult(null);
    prediction.mutate({ data: { features: payload, ...(overview.data?.decisionThreshold != null ? { threshold: overview.data.decisionThreshold } : {}) } }, {
      onSuccess: (response) => setResult(response),
    });
  }
  const errorMessage = prediction.isError ? (prediction.error instanceof Error ? prediction.error.message : 'The scoring request could not be completed. Check the entered fields and try again.') : '';

  return <div className="screening-page rise-in">
    <div className="page-heading screening-heading">
      <div><div className="eyebrow"><span className="eyebrow-line" />INVESTIGATOR WORKFLOW</div><h1 className="font-display">Screen a claim.</h1><p className="page-subtitle">Enter the available claim signals to receive a triage aid.</p></div>
      <div className="workflow-status"><span className="workflow-step"><b>01</b> Enter details</span><span className="workflow-connector" /><span className={`workflow-step ${result ? 'step-done' : ''}`}><b>{result ? <Check size={13} /> : '02'}</b> Review guidance</span></div>
    </div>
    <div className="screening-layout">
      <section className="form-panel">
        <div className="form-panel-header"><div><span className="eyebrow">CLAIM DETAILS</span><h2 className="font-display">What do you know?</h2><p>Fields and valid values are defined by the current model dataset.</p></div><span className="field-count">{catalog.data ? `${catalog.data.featureCount} FIELDS` : 'INPUT'}</span></div>
        {catalog.isLoading && <div className="form-skeleton">{[1, 2, 3, 4].map((n) => <Skeleton key={n} />)}</div>}
        {catalog.isError && <div className="catalog-error"><CircleAlert size={20} /><div><strong>Claim fields are unavailable</strong><p>We can’t safely create a claim form without the supported feature definitions.</p><button onClick={() => catalog.refetch()} className="text-button" data-testid="button-retry-features"><RefreshCw size={14} /> Retry loading fields</button></div></div>}
        {catalog.data && features.length === 0 && <div className="empty-fields"><SlidersHorizontal size={23} /><strong>No supported input fields</strong><p>The feature catalog did not return any fields. No prediction can be submitted.</p></div>}
        {catalog.data && features.length > 0 && <form onSubmit={submit}>
          {Object.entries(grouped).map(([group, groupFeatures]) => <div className="field-group" key={group}><div className="field-group-title">{group}<span>{String(groupFeatures.length).padStart(2, '0')}</span></div><div className="fields-grid">{groupFeatures.map((feature) => <FeatureField key={feature.name} feature={feature} value={values[feature.name] ?? initialValue(feature)} onChange={(v) => setField(feature.name, v)} />)}</div></div>)}
          {validation && <div className="form-message" role="alert"><CircleAlert size={15} />{validation}</div>}
          {prediction.isError && <div className="form-message" role="alert"><CircleAlert size={15} /><span>{errorMessage}</span><button type="button" onClick={() => prediction.reset()} aria-label="Dismiss error"><X size={14} /></button></div>}
          <div className="form-actions"><p><Info size={14} /> Only fields supported by this model are included.</p><button type="submit" className="primary-button submit-button" disabled={prediction.isPending} data-testid="button-score-claim">{prediction.isPending ? <><LoaderCircle size={16} className="spin" /> Scoring claim</> : <>Generate triage guidance <ArrowRight size={15} /></>}</button></div>
        </form>}
      </section>
      <aside className="screening-aside">
        {prediction.isPending ? <div className="pending-panel"><div className="pending-icon"><Activity size={21} /></div><span className="eyebrow">ASSESSMENT IN PROGRESS</span><Skeleton /><Skeleton /><small>Evaluating the supported claim signals.</small></div>
          : result ? <PredictionResult result={result} onDismiss={() => setResult(null)} />
          : <div className="guidance-placeholder">
            <div className="placeholder-mark"><Gauge size={24} /></div><span className="eyebrow">TRIAGE GUIDANCE</span><h2 className="font-display">A signal, not a verdict.</h2><p>Complete the supported claim fields to see a model score and its recommendation.</p><div className="placeholder-rule" /><div className="placeholder-callout"><Shield size={16} /><span>Risk scores help prioritize review. They do not determine whether fraud occurred.</span></div>
          </div>}
        <div className="support-note"><span className="support-icon"><Info size={15} /></span><div><strong>Use judgment alongside the score.</strong><p>Evaluate source documentation and claim context independently.</p></div></div>
      </aside>
    </div>
    <div className="screening-footer"><span><Shield size={14} /> INTERNAL · INVESTIGATOR SUPPORT</span><span>Only dataset-defined claim features are accepted.</span></div>
  </div>;
}

function Router() {
  return <Shell><RoutedErrorBoundary><Switch>
    <Route path="/" component={OverviewPage} />
    <Route path="/screening" component={ScreeningPage} />
    <Route component={NotFound} />
  </Switch></RoutedErrorBoundary></Shell>;
}
function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}
function App() {
  return <QueryClientProvider client={queryClient}><TooltipProvider><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><Router /></WouterRouter><Toaster /></TooltipProvider></QueryClientProvider>;
}

export default App;

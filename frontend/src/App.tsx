import { useState, useEffect, useRef } from 'react';
import './App.css';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend } from 'recharts';
import { Activity, Shield, Terminal, Settings, Server, Cpu, AlertTriangle, CheckCircle2 } from 'lucide-react';

// Types
interface LogEntry {
  type: string;
  text: string;
}

interface ChartData {
  Rows: number;
  'Rolling FNR': number;
  'Rolling FPR': number;
}

interface MissedData {
  Rows: number;
  Strain: string;
  'Missed Samples': number;
}

const X_DOMAIN = [0, 1000000];
const Y_FNR_DOMAIN = [0, 25];
const Y_FPR_DOMAIN = [0, 60];

function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [isRecovering, setIsRecovering] = useState(false);
  
  // Baseline static data
  const [baselineMetrics, setBaselineMetrics] = useState<any>(null);
  const [logisticMetrics, setLogisticMetrics] = useState<any>(null);
  const [staticStreamMetrics, setStaticStreamMetrics] = useState<any>(null);
  const [baselineChartData, setBaselineChartData] = useState<any[]>([]);
  const [baselineMissedData, setBaselineMissedData] = useState<any[]>([]);
  const [baselineExpanderOpen, setBaselineExpanderOpen] = useState(false);
  const [staticExpanderOpen, setStaticExpanderOpen] = useState(false);
  
  // Live Simulation state
  const [metrics, setMetrics] = useState({ acc: 0, prec: 0, rec: 0, f1: 0 });
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [missedData, setMissedData] = useState<MissedData[]>([]);
  const [ensembleSize, setEnsembleSize] = useState(0);
  
  const [socLogs, setSocLogs] = useState<string>('');
  const [agentLogs, setAgentLogs] = useState<string>('');
  
  const [specialists, setSpecialists] = useState<any[]>([]);

  const [expanderOpen, setExpanderOpen] = useState(false);
  const [rosterOpen, setRosterOpen] = useState(false);

  // Simulation Refs
  const logDataRef = useRef<LogEntry[]>([]);
  const pIdxRef = useRef(0);
  const cStatsRef = useRef({ tp: 0, fp: 0, fn: 0, tn: 0 });
  const rollingHistoryRef = useRef<{tp: number, fp: number, fn: number, tn: number}[]>([]);
  const missedTotalRef = useRef<Record<string, number>>({});
  const socLogsBuffer = useRef<string>('');
  const agentLogsBuffer = useRef<string>('');
  
  const socConsoleRef = useRef<HTMLDivElement>(null);
  const agentConsoleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load static data
    fetch('/baseline_metrics_v2.json')
      .then(res => res.json())
      .catch(() => null)
      .then(data => setBaselineMetrics(data));

    fetch('/logistic_metrics.json')
      .then(res => res.json())
      .catch(() => null)
      .then(data => setLogisticMetrics(data));

    // Simulate load_static_baseline_data
    fetch('/static_logs_v2.json')
      .then(res => res.json())
      .catch(() => [])
      .then((data: any[]) => {
        if (!data || data.length === 0) return;
        
        let cumTp = 0, cumFp = 0, cumFn = 0, cumTn = 0;
        const bChart: any[] = [];
        const bMissed: any[] = [];
        const currentMisses: Record<string, number> = {};
        
        const rollFn: number[] = [];
        const rollTp: number[] = [];
        const rollFp: number[] = [];
        const rollTn: number[] = [];
        
        data.forEach(e => {
          if (e.type === 'soc') {
            const match = e.text.match(/\[(\d+)\] .* TP: (\d+) \| FP: (\d+) \| FN: (\d+) \| TN: (\d+)/);
            if (match) {
              const [_, r, tp, fp, fn, tn] = match.map(Number);
              cumTp += tp; cumFp += fp; cumFn += fn; cumTn += tn;
              
              rollFn.push(fn); rollTp.push(tp); rollFp.push(fp); rollTn.push(tn);
              if (rollFn.length > 20) { rollFn.shift(); rollTp.shift(); rollFp.shift(); rollTn.shift(); }
              
              const sumFn = rollFn.reduce((a,b)=>a+b,0);
              const sumTp = rollTp.reduce((a,b)=>a+b,0);
              const sumFp = rollFp.reduce((a,b)=>a+b,0);
              const sumTn = rollTn.reduce((a,b)=>a+b,0);
              
              const rFnr = sumTp+sumFn > 0 ? (sumFn / (sumTp + sumFn)) * 100 : 0;
              const rFpr = sumTn+sumFp > 0 ? (sumFp / (sumTn + sumFp)) * 100 : 0;
              
              bChart.push({ Rows: r, 'Rolling FNR': rFnr, 'Rolling FPR': rFpr });
              
              const atMatch = e.text.match(/Attacks: (\{.*?\}|\[.*?\])/);
              if (atMatch) {
                try {
                  const raw = atMatch[1].replace(/'/g, '"');
                  if (raw.startsWith('{')) {
                    const parsed = JSON.parse(raw);
                    Object.entries(parsed).forEach(([s, count]) => {
                      const name = ["Zero-Day", "DarkNexus"].includes(s) ? "Mimicry Attack" : s;
                      currentMisses[name] = (currentMisses[name] || 0) + Number(count);
                    });
                  } else {
                    const parsed = raw.replace(/[\[\]]/g, '').split(',').map((x: string) => x.trim().replace(/["']/g, ''));
                    parsed.forEach((s: string) => {
                      if (s && s !== 'nan') {
                        const name = ["Zero-Day", "DarkNexus"].includes(s) ? "Mimicry Attack" : s;
                        currentMisses[name] = (currentMisses[name] || 0) + 1;
                      }
                    });
                  }
                } catch(e) {}
                
                Object.entries(currentMisses).forEach(([s, total]) => {
                  bMissed.push({ Rows: r, Strain: s, 'Missed Samples': total });
                });
              }
            }
          }
        });
        
        setBaselineChartData(bChart);
        setBaselineMissedData(bMissed);
        
        const total = cumTp + cumFp + cumFn + cumTn;
        setStaticStreamMetrics({
          accuracy: total > 0 ? (cumTp + cumTn) / total : 0,
          precision: cumTp + cumFp > 0 ? cumTp / (cumTp + cumFp) : 0,
          recall: cumTp + cumFn > 0 ? cumTp / (cumTp + cumFn) : 0,
          f1: cumTp + cumFp > 0 && cumTp + cumFn > 0 ? 
            2 * ((cumTp / (cumTp + cumFp)) * (cumTp / (cumTp + cumFn))) / ((cumTp / (cumTp + cumFp)) + (cumTp / (cumTp + cumFn))) : 0
        });
      });

    fetch('/specialists.json')
      .then(res => res.json())
      .catch(() => [])
      .then(data => setSpecialists(data));
  }, []);

  const anonymizePaths = (text: string) => {
    return text.replace(/\/([a-zA-Z0-9._\-]+\/)+(?=[a-zA-Z0-9._\-]+\.py)/g, "[INTERNAL_PATH]/");
  };

  const startSimulation = () => {
    setIsRunning(true);
    fetch('/logs.json')
      .then(res => res.json())
      .then(data => {
        logDataRef.current = data;
        processNextChunk();
      });
  };

  const processNextChunk = () => {
    if (!isRunning) return;
    
    if (pIdxRef.current >= logDataRef.current.length) {
      setIsRunning(false);
      return;
    }

    const entry = logDataRef.current[pIdxRef.current];
    pIdxRef.current++;

    let shouldPause = false;

    if (entry.type === 'soc') {
      const match = entry.text.match(/\[(\d+)\] .* TP: (\d+) \| FP: (\d+) \| FN: (\d+) \| TN: (\d+)/);
      const ensMatch = entry.text.match(/Ensemble: (\d+)/);
      
      if (match) {
        const [_, r, tp, fp, fn, tn] = match.map(Number);
        cStatsRef.current.tp += tp;
        cStatsRef.current.fp += fp;
        cStatsRef.current.fn += fn;
        cStatsRef.current.tn += tn;
        
        rollingHistoryRef.current.push({tp, fp, fn, tn});
        if (rollingHistoryRef.current.length > 20) rollingHistoryRef.current.shift();
        
        const rTp = rollingHistoryRef.current.reduce((a,b)=>a+b.tp,0);
        const rFp = rollingHistoryRef.current.reduce((a,b)=>a+b.fp,0);
        const rFn = rollingHistoryRef.current.reduce((a,b)=>a+b.fn,0);
        const rTn = rollingHistoryRef.current.reduce((a,b)=>a+b.tn,0);
        
        const rFnr = rTp+rFn > 0 ? (rFn / (rTp + rFn)) * 100 : 0;
        const rFpr = rTn+rFp > 0 ? (rFp / (rTn + rFp)) * 100 : 0;
        
        const total = cStatsRef.current.tp + cStatsRef.current.fp + cStatsRef.current.fn + cStatsRef.current.tn;
        const acc = total > 0 ? (cStatsRef.current.tp + cStatsRef.current.tn) / total : 0;
        const prec = cStatsRef.current.tp + cStatsRef.current.fp > 0 ? cStatsRef.current.tp / (cStatsRef.current.tp + cStatsRef.current.fp) : 0;
        const rec = cStatsRef.current.tp + cStatsRef.current.fn > 0 ? cStatsRef.current.tp / (cStatsRef.current.tp + cStatsRef.current.fn) : 0;
        const f1 = prec + rec > 0 ? 2 * (prec * rec) / (prec + rec) : 0;
        
        setMetrics({ acc, prec, rec, f1 });
        setChartData(prev => [...prev, { Rows: r, 'Rolling FNR': rFnr, 'Rolling FPR': rFpr }]);
        
        const atMatch = entry.text.match(/Attacks: (\{.*?\}|\[.*?\])/);
        if (atMatch) {
          try {
            const raw = atMatch[1].replace(/'/g, '"');
            if (raw.startsWith('{')) {
              const parsed = JSON.parse(raw);
              Object.entries(parsed).forEach(([s, count]) => {
                const name = ["Zero-Day", "DarkNexus"].includes(s) ? "Mimicry Attack" : s;
                missedTotalRef.current[name] = (missedTotalRef.current[name] || 0) + Number(count);
              });
            } else {
              const parsed = raw.replace(/[\[\]]/g, '').split(',').map((x: string) => x.trim().replace(/["']/g, ''));
              parsed.forEach((s: string) => {
                if (s && s !== 'nan') {
                  const name = ["Zero-Day", "DarkNexus"].includes(s) ? "Mimicry Attack" : s;
                  missedTotalRef.current[name] = (missedTotalRef.current[name] || 0) + 1;
                }
              });
            }
            
            const newMissed: MissedData[] = [];
            Object.entries(missedTotalRef.current).forEach(([s, val]) => {
              newMissed.push({ Rows: r, Strain: s, 'Missed Samples': val });
            });
            setMissedData(prev => [...prev, ...newMissed]);
            
          } catch(e) {}
        }
        
        if (ensMatch) setEnsembleSize(Number(ensMatch[1]));
      }

      let cleanSoc = anonymizePaths(entry.text.replace(/Zero-Day/g, "Mimicry Attack"));
      socLogsBuffer.current = (cleanSoc + '\n' + socLogsBuffer.current).substring(0, 2000);
      setSocLogs(socLogsBuffer.current);
      
    } else if (entry.type === 'agent') {
      let txt = anonymizePaths(entry.text);
      if (txt.includes('Warning') || txt.includes('Error')) {
        txt = txt.split('\n')[0] + '\n[SYSTEM] Tuning in progress...';
      }
      agentLogsBuffer.current = (txt + '\n\n' + agentLogsBuffer.current).substring(0, 5000);
      setAgentLogs(agentLogsBuffer.current);
      
    } else if (entry.type === 'critical') {
      
      setIsRecovering(true);
      shouldPause = true;
      setTimeout(() => {
        setIsRecovering(false);
        requestAnimationFrame(processNextChunk);
      }, 5000); // 5 sec pause instead of 15 for better UX in React
    }

    if (!shouldPause) {
      setTimeout(processNextChunk, 50); // Small delay to simulate processing
    }
  };

  useEffect(() => {
    if (isRunning) {
      processNextChunk();
    }
  }, [isRunning]);

  return (
    <div className="app-container">
      <header className="header">
        <h1>Autonomous SOC</h1>
        <p className="text-secondary">Agility and Mitigation of Mimicry Attacks</p>
      </header>

      <div className="panel p-0">
        <div className="expander-header" onClick={() => setExpanderOpen(!expanderOpen)}>
          <div className="panel-title">
            <Server size={20} className="text-accent-blue" />
            Agentic Architecture and Methodology
          </div>
          <span>{expanderOpen ? '▼' : '▶'}</span>
        </div>
        {expanderOpen && (
          <div className="expander-content">
            <h3>Experimental Methodology</h3>
            <p>This research infrastructure demonstrates the fundamental difference between a <strong>Static Detection Perimeter</strong> and an <strong>Agentic Self-Healing SOC</strong>.</p>
            <br />
            <h4>The Mimicry Attack</h4>
            <p>Standard IoT attack patterns (like Mirai or Tsunami) are relatively distinct. However, this simulation injects a <strong>Mimicry Attack</strong> (DarkNexus mutation). This attack is crafted by modifying a malicious syscall pattern so that its frequency distribution in key distinguishing columns mimics benign system behavior.</p>
            <br />
            <h4>Agentic Recovery Pipeline</h4>
            <p>The system utilizes an autonomous feedback loop consisting of two specialized agents:</p>
            <ol style={{ marginLeft: '1.5rem', marginTop: '0.5rem' }}>
              <li><strong>Lead Data Scientist (Architect):</strong> Continuously monitors the ensemble's performance metrics. When a 'Mimicry Breach' is detected, the Architect formulates a new multi-architecture search strategy.</li>
              <li><strong>Execution Engineer (Lab Tech):</strong> Implements the Architect's strategy by writing and executing a custom Scikit-Learn search script.</li>
            </ol>
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Settings size={20} className="text-accent-blue" />
            Simulation Controls
          </div>
        </div>
        <button 
          className="btn-primary" 
          onClick={startSimulation}
          disabled={isRunning || isRecovering}
        >
          {isRunning ? 'Simulation Running...' : 'Start Agentic Detection Simulation'}
        </button>
      </div>

      <div className="panel p-0" style={{ marginBottom: '1rem' }}>
        <div className="expander-header" onClick={() => setBaselineExpanderOpen(!baselineExpanderOpen)}>
          <div className="panel-title">
            <Activity size={20} className="text-accent-blue" />
            Baseline Performance & Training Stats (Pre-Injection)
          </div>
          <span>{baselineExpanderOpen ? '▼' : '▶'}</span>
        </div>
        
        {baselineExpanderOpen && (
          <div className="expander-content">
            {baselineMetrics && (
              <>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>Random Forest Baseline</h4>
                <div className="grid-4" style={{ marginBottom: '1rem' }}>
                  <div className="metric-card">
                    <div className="metric-title" title="The percentage of total predictions that were correct (both malware and benign).">Accuracy (CV)</div>
                    <div className="metric-value">{(baselineMetrics.accuracy.mean * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="Of all samples predicted as malware, how many were actually malware?">Precision (CV)</div>
                    <div className="metric-value">{(baselineMetrics.precision.mean * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="Of all actual malware samples, how many did the model successfully identify?">Recall (CV)</div>
                    <div className="metric-value">{(baselineMetrics.recall.mean * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="The harmonic mean of Precision and Recall, balancing both metrics into a single score.">F1-Score (CV)</div>
                    <div className="metric-value">{(baselineMetrics.f1.mean * 100).toFixed(2)}%</div>
                  </div>
                </div>
                <div className="grid-4" style={{ marginBottom: '1rem' }}>
                  <div className="metric-card">
                    <div className="metric-title" title="The percentage of total predictions that were correct (both malware and benign).">Accuracy (Test Split)</div>
                    <div className="metric-value">{(baselineMetrics.test_accuracy * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="Of all samples predicted as malware, how many were actually malware?">Precision (Test Split)</div>
                    <div className="metric-value">{(baselineMetrics.test_precision * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="Of all actual malware samples, how many did the model successfully identify?">Recall (Test Split)</div>
                    <div className="metric-value">{(baselineMetrics.test_recall * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="The harmonic mean of Precision and Recall, balancing both metrics into a single score.">F1-Score (Test Split)</div>
                    <div className="metric-value">{(baselineMetrics.test_f1 * 100).toFixed(2)}%</div>
                  </div>
                </div>
                <div className="grid-3" style={{ marginBottom: '1rem' }}>
                  <div className="metric-card">
                    <div className="metric-title">Train Split Size</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{baselineMetrics.train_size?.toLocaleString()}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">Test Split Size</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{baselineMetrics.test_size?.toLocaleString()}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">ROC AUC</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{(baselineMetrics.roc_auc * 100).toFixed(2)}%</div>
                  </div>
                </div>
                <div className="grid-2" style={{ marginBottom: '2rem', alignItems: 'center', gap: '2rem' }}>
                  <div style={{ textAlign: 'center' }}>
                    <h5 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>ROC Curve</h5>
                    <img src="/baseline_roc.png" alt="Random Forest ROC Curve" style={{ maxWidth: '100%', borderRadius: '8px', border: '1px solid var(--border-color)' }} />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <h5 style={{ marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Decision Tree Visualization</h5>
                    <a href="https://fabryperez.com/ml-project/deep_baseline_tree.png" target="_blank" rel="noreferrer">
                      <img src="https://fabryperez.com/ml-project/deep_baseline_tree.png" alt="Decision Tree Thumbnail" style={{ maxWidth: '100%', borderRadius: '8px', border: '1px solid var(--border-color)', cursor: 'zoom-in' }} />
                    </a>
                  </div>
                </div>
              </>
            )}
            
            {logisticMetrics && (
              <>
                <h4 style={{ marginBottom: '1rem', color: 'var(--text-secondary)' }}>Logistic Regression Baseline</h4>
                <div className="grid-4" style={{ marginBottom: '1rem' }}>
                  <div className="metric-card">
                    <div className="metric-title">Accuracy (CV)</div>
                    <div className="metric-value">{(logisticMetrics.accuracy.mean * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">Precision (CV)</div>
                    <div className="metric-value">{(logisticMetrics.precision.mean * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">Recall (CV)</div>
                    <div className="metric-value">{(logisticMetrics.recall.mean * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">F1-Score (CV)</div>
                    <div className="metric-value">{(logisticMetrics.f1.mean * 100).toFixed(2)}%</div>
                  </div>
                </div>
                <div className="grid-4" style={{ marginBottom: '1rem' }}>
                  <div className="metric-card">
                    <div className="metric-title">Accuracy (Test Split)</div>
                    <div className="metric-value">{(logisticMetrics.test_accuracy * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">Precision (Test Split)</div>
                    <div className="metric-value">{(logisticMetrics.test_precision * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">Recall (Test Split)</div>
                    <div className="metric-value">{(logisticMetrics.test_recall * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">F1-Score (Test Split)</div>
                    <div className="metric-value">{(logisticMetrics.test_f1 * 100).toFixed(2)}%</div>
                  </div>
                </div>
                <div className="grid-3" style={{ marginBottom: '1rem' }}>
                  <div className="metric-card">
                    <div className="metric-title">Train Split Size</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{logisticMetrics.train_size?.toLocaleString()}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">Test Split Size</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{logisticMetrics.test_size?.toLocaleString()}</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title">ROC AUC</div>
                    <div className="metric-value" style={{ fontSize: '1.2rem' }}>{(logisticMetrics.roc_auc * 100).toFixed(2)}%</div>
                  </div>
                </div>
                <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
                  <img src="/logistic_roc.png" alt="Logistic Regression ROC Curve" style={{ maxWidth: '400px', width: '100%', borderRadius: '8px', border: '1px solid var(--border-color)' }} />
                </div>
              </>
            )}
          </div>
        )}
      </div>


      <div className="panel p-0" style={{ marginBottom: '1rem' }}>
        <div className="expander-header" onClick={() => setStaticExpanderOpen(!staticExpanderOpen)}>
          <div className="panel-title">
            <Activity size={20} className="text-accent-blue" />
            Static Model Performance vs. Mimicry Stream
          </div>
          <span>{staticExpanderOpen ? '▼' : '▶'}</span>
        </div>
        {staticExpanderOpen && (
          <div className="expander-content" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            
            {staticStreamMetrics && (
              <>
                <h4>Cumulative Performance (Million-Row Stream)</h4>
                <div className="grid-4" style={{ marginBottom: '1rem' }}>
                  <div className="metric-card">
                    <div className="metric-title" title="The percentage of total predictions that were correct (both malware and benign).">Accuracy</div>
                    <div className="metric-value">{(staticStreamMetrics.accuracy * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="Of all samples predicted as malware, how many were actually malware?">Precision</div>
                    <div className="metric-value">{(staticStreamMetrics.precision * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="Of all actual malware samples, how many did the model successfully identify?">Recall</div>
                    <div className="metric-value">{(staticStreamMetrics.recall * 100).toFixed(2)}%</div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-title" title="The harmonic mean of Precision and Recall, balancing both metrics into a single score.">F1-Score</div>
                    <div className="metric-value">{(staticStreamMetrics.f1 * 100).toFixed(2)}%</div>
                  </div>
                </div>
              </>
            )}

            {baselineChartData.length > 0 && (
              <>
                <h4>Detection Stability (Rolling Average)</h4>
                <div className="grid-2">
                  <div className="chart-container" style={{ height: '200px' }}>
                    <h5 style={{textAlign:'center', marginBottom: '0.5rem', fontWeight: 600}}>Static Model Rolling FNR</h5>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={baselineChartData}>
                        <defs>
                          <linearGradient id="colorFnrBase" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="Rows" stroke="#9ca3af" domain={X_DOMAIN} type="number" hide />
                        <YAxis stroke="#9ca3af" domain={Y_FNR_DOMAIN} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                        <Area type="monotone" dataKey="Rolling FNR" stroke="#ef4444" fillOpacity={1} fill="url(#colorFnrBase)" isAnimationActive={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="chart-container" style={{ height: '200px' }}>
                    <h5 style={{textAlign:'center', marginBottom: '0.5rem', fontWeight: 600}}>Static Model Rolling FPR</h5>
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={baselineChartData}>
                        <defs>
                          <linearGradient id="colorFprBase" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                        <XAxis dataKey="Rows" stroke="#9ca3af" domain={X_DOMAIN} type="number" hide />
                        <YAxis stroke="#9ca3af" domain={Y_FPR_DOMAIN} />
                        <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                        <Area type="monotone" dataKey="Rolling FPR" stroke="#f59e0b" fillOpacity={1} fill="url(#colorFprBase)" isAnimationActive={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </>
            )}

            {baselineMissedData.length > 0 && (
              <>
                <h4 style={{marginTop: '1rem'}}>Cumulative Missed Samples</h4>
                <div className="chart-container" style={{ height: '300px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={baselineMissedData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="Rows" stroke="#9ca3af" type="number" domain={X_DOMAIN} />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                      <Legend />
                      {Array.from(new Set(baselineMissedData.map(d => d.Strain))).map((strain, i) => (
                        <Line 
                          key={strain} 
                          type="monotone" 
                          dataKey={(d: any) => d.Strain === strain ? d['Missed Samples'] : null} 
                          name={strain} 
                          stroke={['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][i % 5]} 
                          strokeWidth={3} 
                          dot={false}
                          isAnimationActive={false} 
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Shield size={20} className="text-accent-blue" />
            Ensemble Status
          </div>
        </div>
        
        <div className="grid-3" style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1rem' }}>
          <div className="status-badge">
            <div className={`status-dot ${isRunning ? 'active' : ''}`}></div>
            STREAM: {isRunning ? 'ACTIVE' : 'IDLE'}
          </div>
          <div className="status-badge">
            <div className={`status-dot ${isRecovering ? 'danger' : (isRunning ? 'warning' : '')}`}></div>
            AGENT: {isRecovering ? 'RECOVERING' : (isRunning ? 'MONITORING' : 'STANDBY')}
          </div>
          <div className="metric-card" style={{ padding: '0.5rem' }}>
            <span className="metric-title" style={{ marginRight: '1rem' }}>Ensemble Size</span>
            <span className="metric-value" style={{ fontSize: '1.5rem' }}>{ensembleSize}</span>
          </div>
        </div>

        {isRecovering && (
          <div className="recovery-banner">
            <h3><AlertTriangle style={{ display: 'inline', marginBottom: '-4px', marginRight: '8px' }} /> MIMICRY BREACH DETECTED - ORCHESTRATING AGENTIC RECOVERY</h3>
            <p>The Architect is analyzing failure distributions while the Lab Tech runs a multi-architecture search. The simulation will resume as soon as the Specialist is deployed.</p>
          </div>
        )}

        <h4 style={{ margin: '1.5rem 0 1rem', color: 'var(--text-secondary)' }}>Live Ensemble Metrics</h4>
        <div className="grid-4">
          <div className="metric-card">
            <div className="metric-title">Live Accuracy</div>
            <div className={`metric-value ${metrics.acc > 0.95 ? 'good' : 'warning'}`}>{(metrics.acc * 100).toFixed(2)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-title">Live Precision</div>
            <div className={`metric-value ${metrics.prec > 0.9 ? 'good' : 'warning'}`}>{(metrics.prec * 100).toFixed(2)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-title">Live Recall</div>
            <div className={`metric-value ${metrics.rec > 0.9 ? 'good' : (metrics.rec < 0.7 && metrics.rec > 0 ? 'danger' : 'warning')}`}>{(metrics.rec * 100).toFixed(2)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-title">Live F1-Score</div>
            <div className={`metric-value ${metrics.f1 > 0.9 ? 'good' : 'warning'}`}>{(metrics.f1 * 100).toFixed(2)}%</div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <div className="panel-title mb-4">Real-Time False Negative Rate</div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorFnr" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="Rows" stroke="#9ca3af" domain={X_DOMAIN} type="number" />
                <YAxis stroke="#9ca3af" domain={Y_FNR_DOMAIN} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                <Area type="monotone" dataKey="Rolling FNR" stroke="#ef4444" fillOpacity={1} fill="url(#colorFnr)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title mb-4">Real-Time False Positive Rate</div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorFpr" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="Rows" stroke="#9ca3af" domain={X_DOMAIN} type="number" />
                <YAxis stroke="#9ca3af" domain={Y_FPR_DOMAIN} />
                <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
                <Area type="monotone" dataKey="Rolling FPR" stroke="#f59e0b" fillOpacity={1} fill="url(#colorFpr)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title mb-4">Cumulative Missed Attack Samples</div>
        <div className="chart-container" style={{ height: '400px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={missedData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="Rows" stroke="#9ca3af" type="number" domain={X_DOMAIN} />
              <YAxis stroke="#9ca3af" />
              <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151' }} />
              <Legend />
              {/* Group data by strain dynamically */}
              {Array.from(new Set(missedData.map(d => d.Strain))).map((strain, i) => (
                <Line 
                  key={strain} 
                  type="monotone" 
                  dataKey={(d: any) => d.Strain === strain ? d['Missed Samples'] : null} 
                  name={strain} 
                  stroke={['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'][i % 5]} 
                  strokeWidth={3} 
                  dot={false}
                  isAnimationActive={false} 
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel p-0">
        <div className="expander-header" onClick={() => setRosterOpen(!rosterOpen)}>
          <div className="panel-title">
            <CheckCircle2 size={20} className="text-accent-green" />
            Specialist Agent Roster
          </div>
          <span>{rosterOpen ? '▼' : '▶'}</span>
        </div>
        {rosterOpen && (
          <div className="expander-content grid-2">
            {specialists.length > 0 ? (
              specialists.map((spec, idx) => (
                <div key={idx} className="roster-card">
                  <h4>{spec.name}</h4>
                  <p className="text-secondary" style={{ fontSize: '0.8rem', marginBottom: '0.5rem' }}>
                    Deployed: {spec.timestamp.replace(/_/g, ' ')}
                  </p>
                  <div style={{ marginBottom: '0.5rem' }}>
                    <span className="text-secondary" style={{ fontSize: '0.8rem' }}>Train Size: {spec.train_size} | Test Size: {spec.test_size} | Acc: {(spec.accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <div className="roster-json">
                    {JSON.stringify(spec.params, null, 2)}
                  </div>
                </div>
              ))
            ) : (
              <div className="roster-card" style={{ gridColumn: '1 / -1', textAlign: 'center', opacity: 0.7 }}>
                No specialists deployed yet.
              </div>
            )}
          </div>
        )}
      </div>

      <div className="grid-2">
        <div className="panel">
          <div className="panel-title mb-4">
            <Terminal size={20} className="text-accent-blue" />
            SOC Processing Stream
          </div>
          <div className="console-container" ref={socConsoleRef}>
            <pre>{socLogs}</pre>
          </div>
        </div>
        <div className="panel">
          <div className="panel-title mb-4">
            <Cpu size={20} className="text-accent-green" />
            Agent Thought Process
          </div>
          <div className="console-container agent" ref={agentConsoleRef}>
            <pre>{agentLogs}</pre>
          </div>
        </div>
      </div>

    </div>
  );
}

export default App;

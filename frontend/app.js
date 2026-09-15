const { useState, useEffect, useCallback, useRef } = React;
const API_BASE = window.API_BASE;

/* ==========================================================================
   API helpers
   ========================================================================== */
async function apiGet(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}
async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

/* ==========================================================================
   Domain formatting. Thresholds (15min / 45min) are deliberately the SAME
   numbers the backend already uses (calibrate.py's on-time definition and
   domain_constants.CREW_CHANGEOVER_RISK_THRESHOLD_MIN) -- not separate
   UI-only cutoffs invented for color-coding.
   ========================================================================== */
function delayStatus(min) {
  if (min <= 15) return "ontime";
  if (min <= 45) return "watch";
  return "critical";
}
function fmtDelay(min) {
  const sign = min >= 0 ? "+" : "";
  return `${sign}${min.toFixed(1)}m`;
}

const DEFAULT_LIVE_STATE = {
  "12301": { current_station: "GAYA", current_delay_min: 15, on_date: "2026-01-05" },
  "12951": { current_station: "RTM", current_delay_min: 10, on_date: "2026-03-10" },
  "10103": { current_station: "ROHA", current_delay_min: 10, on_date: "2026-07-14" },
  "10104": { current_station: "RN", current_delay_min: 5, on_date: "2026-07-14" },
  "12013": { current_station: "UMB", current_delay_min: 8, on_date: "2026-01-05" },
};

const POLL_INTERVAL_MS = 4000;

const TAB_DEFS = [
  { id: "fleet", label: "Fleet" },
  { id: "detail", label: "Train" },
  { id: "network", label: "Network" },
  { id: "feed", label: "Live Feed" },
  { id: "trust", label: "Trust" },
];

/* ==========================================================================
   Rail
   ========================================================================== */
function Rail({ active, onChange, apiOk }) {
  const handleKey = (e, id) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onChange(id);
    }
  };

  return (
    <div className="rail">
      <div className="rail-brand"><strong>ETA</strong> CONTROL</div>
      <div role="tablist" aria-label="Sections" style={{ display: "contents" }}>
        {TAB_DEFS.map((t) => (
          <div
            key={t.id}
            className={"rail-tab" + (active === t.id ? " active" : "")}
            onClick={() => onChange(t.id)}
            onKeyDown={(e) => handleKey(e, t.id)}
            role="tab"
            tabIndex={0}
            aria-selected={active === t.id}
          >
            <span className="dot" style={{ background: apiOk ? "var(--signal-green)" : "var(--signal-red)" }} />
            {t.label}
          </div>
        ))}
      </div>
      <div className="rail-footer">
        api: {apiOk ? "connected" : "unreachable"}<br />{API_BASE}
      </div>
    </div>
  );
}

/* ==========================================================================
   Fleet view
   ========================================================================== */
function FleetView({ trainMeta, predictions, onSelect, justChanged }) {
  const trainNos = Object.keys(DEFAULT_LIVE_STATE);

  return (
    <div>
      <h1>Fleet</h1>
      <p className="subtitle">Live-tracked trains, auto-refreshing every {POLL_INTERVAL_MS / 1000}s.</p>
      <div className="panel">
        <table className="board">
          <thead>
            <tr>
              <th>Train</th>
              <th>Route</th>
              <th>Next station</th>
              <th>Delay at next stop</th>
              <th>At destination</th>
              <th>Cause</th>
            </tr>
          </thead>
          <tbody>
            {trainNos.map((tno) => {
              const pred = predictions[tno];
              const meta = trainMeta[tno];
              const first = pred && pred.stations && pred.stations[0];
              const last = pred && pred.stations && pred.stations[pred.stations.length - 1];
              const flip = justChanged[tno] && justChanged[tno].has(first && first.station);
              return (
                <tr key={tno} className="clickable" data-testid={`fleet-row-${tno}`} onClick={() => onSelect(tno)}>
                  <td><span className="train-no">{tno}</span></td>
                  <td>{meta ? meta.name : "..."}</td>
                  <td>{first ? `${first.station} @ ${first.eta_p50}` : "--"}</td>
                  <td>
                    {first ? (
                      <span className={`delay-figure status-${delayStatus(first.predicted_delay_min)}${flip ? " flip" : ""}`}>
                        {fmtDelay(first.predicted_delay_min)}
                      </span>
                    ) : "--"}
                  </td>
                  <td>
                    {last ? (
                      <span className={`mono status-${delayStatus(last.predicted_delay_min)}`}>
                        {last.station} {fmtDelay(last.predicted_delay_min)}
                      </span>
                    ) : "--"}
                  </td>
                  <td>
                    {first && first.cause_tags.map((c) => (
                      <span key={c} className={`badge tag-${c}`}>{c.replace(/_/g, " ")}</span>
                    ))}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ==========================================================================
   Train detail view: simulator controls + timeline + propagation notes
   ========================================================================== */
function Simulator({ trainNo, trainMeta, liveState, onApply }) {
  const meta = trainMeta[trainNo];
  const state = liveState[trainNo] || {};
  const [station, setStation] = useState(state.current_station || "");
  const [delay, setDelay] = useState(state.current_delay_min || 0);
  const [onDate, setOnDate] = useState(state.on_date || "2026-01-05");

  useEffect(() => {
    setStation(state.current_station || "");
    setDelay(state.current_delay_min || 0);
    setOnDate(state.on_date || "2026-01-05");
  }, [trainNo]);

  if (!meta) return null;
  const stationOptions = meta.stations.slice(0, -1); // can't be "at" the final destination

  return (
    <div className="panel">
      <div className="panel-title">Scenario simulator -- inject a live position/delay and watch the pipeline react</div>
      <div className="controls-row">
        <div>
          <label>Current station</label>
          <select data-testid="station-select" value={station} onChange={(e) => setStation(e.target.value)}>
            {stationOptions.map((s) => <option key={s.code} value={s.code}>{s.code} -- {s.name}</option>)}
          </select>
        </div>
        <div>
          <label>Current delay (min)</label>
          <input type="number" value={delay} step="5" onChange={(e) => setDelay(parseFloat(e.target.value) || 0)} />
        </div>
        <div>
          <label>Date</label>
          <input type="date" value={onDate} onChange={(e) => setOnDate(e.target.value)} />
        </div>
        <button onClick={() => onApply(trainNo, { current_station: station, current_delay_min: delay, on_date: onDate })}>
          Apply
        </button>
      </div>
    </div>
  );
}

function StationTimeline({ stations }) {
  return (
    <div className="timeline">
      {stations.map((s) => (
        <div className="tl-station" key={s.station}>
          <div className="tl-band" />
          <div className="tl-code">{s.station}</div>
          <div className="tl-name">sched {s.sched_arr || "--"}</div>
          <div className={`tl-delay status-${delayStatus(s.predicted_delay_min)}`}>{fmtDelay(s.predicted_delay_min)}</div>
          <div className="tl-band-range">{s.eta_p10} - {s.eta_p50} - {s.eta_p90}</div>
          <div className="tl-note">{s.confidence_note}</div>
          {s.top_reasons.length > 0 && (
            <div className="reasons-list">
              {s.top_reasons.map((r, i) => (
                <div className="reason-row" key={i}>
                  <span>{r.factor}</span>
                  <span className={`reason-impact ${r.impact_min >= 0 ? "pos" : "neg"}`}>{fmtDelay(r.impact_min)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function PropagationNotes({ prediction }) {
  const sections = [
    { key: "crew_changeover_notes", label: "Crew changeover", critical: true },
    { key: "single_line_self_hold_notes", label: "Single-line hold (this train)", critical: true },
    { key: "single_line_opposing_train_notes", label: "Single-line hold (opposing train)", critical: false },
    { key: "hub_connection_notes", label: "Hub connection", critical: false },
  ];
  const any = sections.some((s) => (prediction[s.key] || []).length > 0);
  if (!any) return <div className="empty-state">No propagation effects triggered for this scenario.</div>;

  return (
    <div>
      {sections.map((s) => (prediction[s.key] || []).map((n, i) => (
        <div className={`notice${s.critical ? " critical" : ""}`} key={`${s.key}-${i}`}>
          <div className="notice-type">{s.label}</div>
          {n.note}
        </div>
      )))}
      {(prediction.chained_connections || []).map((c, i) => (
        <div className="notice" key={`chain-${i}`}>
          <div className="notice-type">Chained: {c.connecting_train} inherits ~{c.inherited_delay_min}min via {c.via_hub}</div>
          {c.prediction && c.prediction.stations.slice(0, 3).map((s) => (
            <span key={s.station} className="mono" style={{ marginRight: 14 }}>
              {s.station} {fmtDelay(s.predicted_delay_min)}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function TrainDetailView({ trainNo, setTrainNo, trainMeta, prediction, liveState, onApply }) {
  const trainNos = Object.keys(DEFAULT_LIVE_STATE);
  return (
    <div>
      <h1>Train detail</h1>
      <p className="subtitle">Full station-by-station prediction with uncertainty band, root cause, and cascade effects.</p>

      <div className="controls-row" style={{ marginBottom: 16 }}>
        <div>
          <label>Train</label>
          <select data-testid="train-select" value={trainNo} onChange={(e) => setTrainNo(e.target.value)}>
            {trainNos.map((t) => <option key={t} value={t}>{t} -- {trainMeta[t] ? trainMeta[t].name : ""}</option>)}
          </select>
        </div>
      </div>

      <Simulator trainNo={trainNo} trainMeta={trainMeta} liveState={liveState} onApply={onApply} />

      {prediction ? (
        <React.Fragment>
          <div className="panel">
            <div className="panel-title">Station timeline -- season: {prediction.season}</div>
            <StationTimeline stations={prediction.stations} />
          </div>
          <div className="panel">
            <div className="panel-title">Propagation &amp; cascade effects</div>
            <PropagationNotes prediction={prediction} />
          </div>
        </React.Fragment>
      ) : <div className="empty-state">Loading prediction...</div>}
    </div>
  );
}

/* ==========================================================================
   Network view -- schematic diagram, not a geographic map (no real GPS
   coordinates exist in this dataset -- said explicitly rather than
   implying accuracy that isn't there).
   ========================================================================== */
function NetworkView({ networkData }) {
  if (!networkData) return <div className="empty-state">Loading network topology...</div>;

  const laneHeight = 90;
  const stationGap = 150;
  const leftPad = 90;
  const width = 1100;

  const sectionsByTrain = {};
  (networkData.sections || []).forEach((s) => {
    sectionsByTrain[`${s.train_no}|${s.from}|${s.to}`] = s;
  });

  return (
    <div>
      <h1>Network</h1>
      <p className="subtitle">Schematic route diagram (not geographic) -- amber dashed edges are single-line block sections; red edges are crew-changeover points.</p>
      <div className="panel">
        <svg className="network-svg" viewBox={`0 0 ${width} ${networkData.trains.length * laneHeight + 40}`}>
          {networkData.trains.map((t, ti) => {
            const y = ti * laneHeight + 50;
            return (
              <g key={t.train_no}>
                <text x="10" y={y + 5} className="net-station-label" fill="var(--amber)">{t.train_no}</text>
                {t.stations.map((s, si) => {
                  if (si === 0) return null;
                  const prev = t.stations[si - 1];
                  const sec = sectionsByTrain[`${t.train_no}|${prev.code}|${s.code}`];
                  const x1 = leftPad + (si - 1) * stationGap;
                  const x2 = leftPad + si * stationGap;
                  let cls = "net-edge";
                  if (sec && sec.single_line) cls += " single-line";
                  if (sec && sec.crew_changeover) cls += " crew";
                  return <line key={s.code} x1={x1} y1={y} x2={x2} y2={y} className={cls} />;
                })}
                {t.stations.map((s, si) => {
                  const x = leftPad + si * stationGap;
                  return (
                    <g key={s.code}>
                      <circle cx={x} cy={y} r="6" className="net-station-dot" />
                      <text x={x} y={y - 12} textAnchor="middle" className="net-station-label">{s.code}</text>
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
      </div>
      <div className="panel">
        <div className="panel-title">Connecting trains (hub propagation)</div>
        {(networkData.connecting_trains || []).map((c, i) => (
          <div key={i} className="notice">{c.arriving_train} &rarr; {c.connection} at {c.hub_station}. {c.note}</div>
        ))}
      </div>
    </div>
  );
}

/* ==========================================================================
   Live feed view
   ========================================================================== */
function LiveFeedView({ status, onTick, ticking }) {
  const log = status ? [...status.recent_log].reverse() : [];
  return (
    <div>
      <h1>Live feed</h1>
      <p className="subtitle">Real events from the backend's autonomous dispatch loop (30s interval), or trigger one now.</p>

      <div className="panel">
        <div className="controls-row">
          <button onClick={onTick} disabled={ticking}>{ticking ? "Running..." : "Trigger dispatch tick now"}</button>
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">Tracked trains</div>
        {status && Object.keys(status.tracked_trains).map((tno) => {
          const s = status.tracked_trains[tno];
          return (
            <div key={tno} className="stat-row">
              <span className="stat-label mono">{tno}</span>
              <span className="stat-value">{s.current_station} / {fmtDelay(s.current_delay_min)}</span>
            </div>
          );
        })}
      </div>

      <div className="panel">
        <div className="panel-title">Subscribers</div>
        {status && status.subscribers.length === 0 && <div className="empty-state">No webhook subscribers registered.</div>}
        {status && status.subscribers.map((s, i) => (
          <div key={i} className="stat-row"><span className="mono">{s.url}</span><span className="text-muted">{s.train_no || "all trains"}</span></div>
        ))}
      </div>

      <div className="panel">
        <div className="panel-title">Event log</div>
        {log.length === 0 && <div className="feed-empty">No events fired yet -- apply a scenario change on the Train tab, then wait or trigger a tick.</div>}
        {log.map((e, i) => (
          <div className="feed-row" key={i}>
            <span className="feed-time">{(e.timestamp || (e.payload && e.payload.timestamp) || "").slice(11, 19)}</span>
            <span>
              {e.error ? `ERROR: ${e.error}` :
                `${e.train_no} / ${e.station}  ${fmtDelay(e.delta_min)}  -> fired to ${e.fired_to.length} subscriber(s)`}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ==========================================================================
   Trust view
   ========================================================================== */
function TrustView({ metrics }) {
  if (!metrics) return <div className="empty-state">Loading model evaluation...</div>;
  return (
    <div>
      <h1>Trust</h1>
      <p className="subtitle">Real evaluation results from the last training run, on a held-out stratified test split -- not cherry-picked.</p>

      <div className="panel">
        <div className="panel-title">Overall (n={metrics.n_test_rows} test rows)</div>
        <div className="stat-row"><span className="stat-label">Baseline-only MAE</span><span className="stat-value">{metrics.baseline_mae.toFixed(2)} min</span></div>
        <div className="stat-row"><span className="stat-label">Baseline + ML correction MAE</span><span className="stat-value good">{metrics.model_mae.toFixed(2)} min</span></div>
        <div className="stat-row"><span className="stat-label">Improvement</span><span className="stat-value good">+{metrics.improvement_pct.toFixed(1)}%</span></div>
        <div className="stat-row"><span className="stat-label">p10-p90 coverage (target ~80%)</span><span className="stat-value">{(metrics.coverage_80 * 100).toFixed(1)}%</span></div>
      </div>

      <div className="panel">
        <div className="panel-title">Where the model earns its keep -- MAE improvement by season</div>
        <table className="season-table">
          <thead><tr><th>Season</th><th>n</th><th>Baseline MAE</th><th>Model MAE</th><th>Improvement</th></tr></thead>
          <tbody>
            {metrics.season_breakdown.map((s) => (
              <tr key={s.season}>
                <td>{s.season}</td><td>{s.n}</td><td>{s.baseline_mae}</td><td>{s.model_mae}</td>
                <td className={s.improvement_pct > 3 ? "status-ontime" : ""}>+{s.improvement_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ==========================================================================
   App
   ========================================================================== */
function App() {
  const [tab, setTab] = useState("fleet");
  const [apiOk, setApiOk] = useState(true);
  const [trainMeta, setTrainMeta] = useState({});
  const [networkData, setNetworkData] = useState(null);
  const [liveState, setLiveState] = useState(DEFAULT_LIVE_STATE);
  const [predictions, setPredictions] = useState({});
  const [selectedTrain, setSelectedTrain] = useState("12301");
  const [dispatchStatus, setDispatchStatus] = useState(null);
  const [evalMetrics, setEvalMetrics] = useState(null);
  const [ticking, setTicking] = useState(false);
  const [justChanged, setJustChanged] = useState({});

  const prevDelaysRef = useRef({});

  const fetchPrediction = useCallback(async (tno, state) => {
    try {
      const params = new URLSearchParams({
        current_station: state.current_station,
        current_delay_min: state.current_delay_min,
        on_date: state.on_date,
      });
      const pred = await apiGet(`/eta/${tno}?${params.toString()}`);
      setApiOk(true);

      const prevMap = prevDelaysRef.current[tno] || {};
      const changed = new Set();
      pred.stations.forEach((s) => {
        const prevVal = prevMap[s.station];
        if (prevVal !== undefined && Math.abs(prevVal - s.predicted_delay_min) > 0.05) changed.add(s.station);
        prevMap[s.station] = s.predicted_delay_min;
      });
      prevDelaysRef.current[tno] = prevMap;
      if (changed.size > 0) {
        setJustChanged((jc) => ({ ...jc, [tno]: changed }));
        setTimeout(() => setJustChanged((jc) => ({ ...jc, [tno]: new Set() })), 500);
      }

      setPredictions((p) => ({ ...p, [tno]: pred }));
    } catch (e) {
      setApiOk(false);
    }
  }, []);

  const refreshAll = useCallback(() => {
    Object.keys(liveState).forEach((tno) => fetchPrediction(tno, liveState[tno]));
  }, [liveState, fetchPrediction]);

  const refreshDispatchStatus = useCallback(async () => {
    try {
      const s = await apiGet("/dispatch/status");
      setDispatchStatus(s);
    } catch (e) { /* handled by apiOk elsewhere */ }
  }, []);

  const handleApply = useCallback(async (tno, newState) => {
    setLiveState((ls) => ({ ...ls, [tno]: newState }));
    try {
      await apiPost("/dispatch/track", { train_no: tno, ...newState });
      setApiOk(true);
      fetchPrediction(tno, newState);
    } catch (e) {
      setApiOk(false);
    }
  }, [fetchPrediction]);

  const handleTick = useCallback(async () => {
    setTicking(true);
    try {
      await apiPost("/dispatch/tick", {});
      setApiOk(true);
      await refreshDispatchStatus();
    } catch (e) {
      setApiOk(false);
    } finally {
      setTicking(false);
    }
  }, [refreshDispatchStatus]);

  // Initial load: network topology, eval metrics, register all fleet trains
  // for real dispatch tracking, first prediction pass.
  useEffect(() => {
    (async () => {
      try {
        const net = await apiGet("/network");
        const meta = {};
        net.trains.forEach((t) => { meta[t.train_no] = t; });
        setTrainMeta(meta);
        setNetworkData(net);
        setApiOk(true);
      } catch (e) { setApiOk(false); }

      try { setEvalMetrics(await apiGet("/model/eval_metrics")); } catch (e) {}

      for (const tno of Object.keys(DEFAULT_LIVE_STATE)) {
        try { await apiPost("/dispatch/track", { train_no: tno, ...DEFAULT_LIVE_STATE[tno] }); } catch (e) {}
      }
      // Establish a baseline dispatch snapshot immediately. Without this,
      // the FIRST tick anyone ever triggers (manual or the backend's own
      // autonomous 30s loop) always fires zero events -- not a bug, just
      // an inherent property of diff-based change detection with nothing
      // yet to diff against (see BUILD_LOG.md). Ticking once here means a
      // scenario change applied afterward has a real prior value to be
      // compared to, so the Live Feed tab can actually show something on
      // the very next tick instead of only ever showing it on the third.
      try { await apiPost("/dispatch/tick", {}); } catch (e) {}
      refreshAll();
      refreshDispatchStatus();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll loop.
  useEffect(() => {
    const id = setInterval(() => {
      refreshAll();
      refreshDispatchStatus();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [refreshAll, refreshDispatchStatus]);

  return (
    <React.Fragment>
      <Rail active={tab} onChange={setTab} apiOk={apiOk} />
      <div className="content">
        {tab === "fleet" && (
          <FleetView trainMeta={trainMeta} predictions={predictions}
            onSelect={(tno) => { setSelectedTrain(tno); setTab("detail"); }} justChanged={justChanged} />
        )}
        {tab === "detail" && (
          <TrainDetailView trainNo={selectedTrain} setTrainNo={setSelectedTrain} trainMeta={trainMeta}
            prediction={predictions[selectedTrain]} liveState={liveState} onApply={handleApply} />
        )}
        {tab === "network" && <NetworkView networkData={networkData} />}
        {tab === "feed" && <LiveFeedView status={dispatchStatus} onTick={handleTick} ticking={ticking} />}
        {tab === "trust" && <TrustView metrics={evalMetrics} />}
      </div>
    </React.Fragment>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
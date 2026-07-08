import { useEffect, useMemo, useState } from "react";
import type { ParticipantResult } from "../types";
import { severityColor, severityLabel } from "../utils";
import { FusionStrip } from "./FusionStrip";

interface ParticipantTableProps {
  results: ParticipantResult[];
  attributions: Record<number, Record<string, number>>;
  selectedPid: number | null;
  onSelect: (pid: number) => void;
}

type SortKey = "participant_id" | "n_segments" | "phq8_true" | "phq8_pred";
type SortDir = "asc" | "desc";

const SEVERITY_BANDS = ["Minimal", "Mild", "Moderate", "Mod. severe", "Severe"];
const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];

export function ParticipantTable({ results, attributions, selectedPid, onSelect }: ParticipantTableProps) {
  const [search, setSearch] = useState("");
  const [splitFilter, setSplitFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("participant_id");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  const availableSplits = useMemo(
    () => Array.from(new Set(results.map((r) => r.split))).sort(),
    [results]
  );

  // Any filter change invalidates the current page — otherwise you can land
  // on "page 4 of 1" after narrowing the result set.
  useEffect(() => {
    setPage(1);
  }, [search, splitFilter, severityFilter, pageSize]);

  const filtered = useMemo(() => {
    let rows = results;
    const q = search.trim().toLowerCase().replace(/^p/, "");
    if (q) {
      rows = rows.filter((r) => String(r.participant_id).includes(q));
    }
    if (splitFilter !== "all") {
      rows = rows.filter((r) => r.split === splitFilter);
    }
    if (severityFilter !== "all") {
      // Filters on TRUE severity, not predicted — with mock encoders every
      // prediction currently lands in "Minimal" regardless of ground truth,
      // so filtering on predicted severity wouldn't isolate anything useful
      // yet. True-severity filtering already lets you inspect e.g. "how
      // does the model currently handle the Severe ground-truth cases".
      rows = rows.filter((r) => severityLabel(r.phq8_true) === severityFilter);
    }
    return rows;
  }, [results, search, splitFilter, severityFilter]);

  const sorted = useMemo(() => {
    const copy = [...filtered];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [filtered, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const clampedPage = Math.min(page, totalPages);
  const pageRows = sorted.slice((clampedPage - 1) * pageSize, clampedPage * pageSize);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  function sortIndicator(key: SortKey) {
    if (sortKey !== key) return null;
    return <span className="sort-arrow">{sortDir === "asc" ? "▲" : "▼"}</span>;
  }

  const hasFilters = search !== "" || splitFilter !== "all" || severityFilter !== "all";

  return (
    <div>
      <div className="table-toolbar">
        <input
          type="text"
          className="table-search"
          placeholder="Search participant ID…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="table-select"
          value={splitFilter}
          onChange={(e) => setSplitFilter(e.target.value)}
          aria-label="Filter by split"
        >
          <option value="all">All splits</option>
          {availableSplits.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select
          className="table-select"
          value={severityFilter}
          onChange={(e) => setSeverityFilter(e.target.value)}
          aria-label="Filter by true severity"
        >
          <option value="all">All severities (true)</option>
          {SEVERITY_BANDS.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
        {hasFilters && (
          <button
            className="table-clear"
            onClick={() => {
              setSearch("");
              setSplitFilter("all");
              setSeverityFilter("all");
            }}
          >
            Clear
          </button>
        )}
        <span className="table-result-count">
          {sorted.length} of {results.length}
        </span>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort("participant_id")}>
                Participant {sortIndicator("participant_id")}
              </th>
              <th>Split</th>
              <th className="sortable" onClick={() => toggleSort("n_segments")}>
                Segments {sortIndicator("n_segments")}
              </th>
              <th className="sortable" onClick={() => toggleSort("phq8_true")}>
                True PHQ-8 {sortIndicator("phq8_true")}
              </th>
              <th className="sortable" onClick={() => toggleSort("phq8_pred")}>
                Pred PHQ-8 {sortIndicator("phq8_pred")}
              </th>
              <th>Severity (pred)</th>
              <th>Modality signal</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 && (
              <tr>
                <td colSpan={7} className="table-empty">
                  No participants match these filters.
                </td>
              </tr>
            )}
            {pageRows.map((r) => {
              const color = severityColor(r.phq8_pred);
              return (
                <tr
                  key={r.participant_id}
                  className={r.participant_id === selectedPid ? "selected" : ""}
                  onClick={() => onSelect(r.participant_id)}
                >
                  <td className="pid-cell">P{r.participant_id}</td>
                  <td>
                    <span className="split-tag">{r.split}</span>
                  </td>
                  <td className="num dim">{r.n_segments}</td>
                  <td className="num">{r.phq8_true}</td>
                  <td className="num">{r.phq8_pred.toFixed(2)}</td>
                  <td>
                    <span className="severity-chip" style={{ color }}>
                      <span className="dot" style={{ background: color }} />
                      {severityLabel(r.phq8_pred)}
                    </span>
                  </td>
                  <td style={{ width: 90 }}>
                    <FusionStrip attributions={attributions[r.participant_id] ?? {}} size="sm" />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {sorted.length > 0 && (
        <div className="table-pagination">
          <div className="pagination-info">
            Showing {(clampedPage - 1) * pageSize + 1}–{Math.min(clampedPage * pageSize, sorted.length)} of{" "}
            {sorted.length}
          </div>
          <div className="pagination-controls">
            <select
              className="table-select sm"
              value={pageSize}
              onChange={(e) => setPageSize(Number(e.target.value))}
              aria-label="Rows per page"
            >
              {PAGE_SIZE_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n} / page
                </option>
              ))}
            </select>
            <button className="page-btn" disabled={clampedPage <= 1} onClick={() => setPage((p) => p - 1)}>
              ‹ Prev
            </button>
            <span className="page-indicator">
              Page {clampedPage} of {totalPages}
            </span>
            <button
              className="page-btn"
              disabled={clampedPage >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next ›
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

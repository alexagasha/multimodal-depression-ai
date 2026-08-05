"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, ApiError, PatientSummary, VisitSummary, TreatmentEvent } from "@/lib/api";
import { Card } from "@/components/FormField";
import TrendSparkline from "@/components/TrendSparkline";
import TreatmentEvents from "@/components/TreatmentEvents";

const DEMOGRAPHIC_FIELDS: { key: keyof PatientSummary; label: string }[] = [
  { key: "age_band", label: "Age band" },
  { key: "sex", label: "Sex" },
  { key: "marital_status", label: "Marital status" },
  { key: "ethnicity", label: "Ethnicity / tribe" },
  { key: "residence", label: "Residence" },
  { key: "education_level", label: "Education level" },
  { key: "employment_status", label: "Employment status" },
  { key: "smartphone", label: "Owns a smartphone?" },
  { key: "site", label: "Site" },
];

function humanize(value: unknown, key?: keyof PatientSummary): string {
  if (typeof value !== "string" || !value) return "–";
  // age_band is a range ("26-35") — the hyphen is meaningful, not a word
  // separator like in "self-employed" or "semi-urban", so leave it alone.
  const spaced = key === "age_band" ? value.replace(/_/g, " ") : value.replace(/[_-]/g, " ");
  return spaced.replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function PatientDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [patient, setPatient] = useState<PatientSummary | null>(null);
  const [visits, setVisits] = useState<VisitSummary[] | null>(null);
  const [treatmentEvents, setTreatmentEvents] = useState<TreatmentEvent[]>([]);
  const [notFound, setNotFound] = useState(false);
  const [startingVisit, setStartingVisit] = useState(false);

  useEffect(() => {
    api
      .getPatient(id)
      .then(setPatient)
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) setNotFound(true);
      });
    api.listPatientVisits(id).then(setVisits).catch(() => {});
  }, [id]);

  async function startNewVisit() {
    setStartingVisit(true);
    try {
      const { session_id } = await api.startVisit(id);
      router.push(`/visits/${session_id}`);
    } finally {
      setStartingVisit(false);
    }
  }

  if (notFound) return <p className="text-sm text-sage-700">Patient {id} not found.</p>;
  if (!patient || !visits) return <p className="text-sm text-sage-700">Loading…</p>;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="font-display text-2xl font-semibold text-sage-900">
            Patient {id}
          </h1>
          <p className="text-sm text-sage-700">
            {visits.length} visit{visits.length === 1 ? "" : "s"} on record
            {patient.last_visit_at && ` · last seen ${new Date(patient.last_visit_at).toLocaleDateString()}`}
          </p>
        </div>
        <button
          onClick={startNewVisit}
          disabled={startingVisit}
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-sage-600 disabled:opacity-50"
        >
          {startingVisit ? "Starting…" : "+ Start new visit"}
        </button>
      </div>

      {patient.risk_flag && (
        <div
          role="alert"
          className="rounded-xl border-2 border-[var(--color-danger-border)] bg-[var(--color-danger-bg)] px-4 py-3 font-semibold text-[var(--color-danger)]"
        >
          ⚠ This patient&apos;s most recent visit has an active referral flag.
        </div>
      )}

      {(patient.relapse_warning.flag || patient.risk_trajectory.flag) && (
        <div className="space-y-2">
          {patient.relapse_warning.flag && (
            <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <span className="font-semibold">Worsening trend detected —</span>{" "}
              {patient.relapse_warning.reason}
            </div>
          )}
          {patient.risk_trajectory.flag && (
            <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              <span className="font-semibold">Rising risk pattern —</span>{" "}
              {patient.risk_trajectory.reason}
            </div>
          )}
        </div>
      )}

      <Card>
        <h2 className="font-display text-base font-semibold text-sage-800">Demographics</h2>
        <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-3">
          {DEMOGRAPHIC_FIELDS.map(({ key, label }) => (
            <div key={key}>
              <dt className="text-xs text-sage-600">{label}</dt>
              <dd className="text-ink-900">{humanize(patient[key], key)}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <Card>
        <h2 className="font-display text-base font-semibold text-sage-800">
          Severity over time
        </h2>
        <div className="mt-3">
          <TrendSparkline visits={visits} events={treatmentEvents} />
        </div>
      </Card>

      <TreatmentEvents patientId={id} onChange={setTreatmentEvents} />

      <ul className="space-y-2">
        {visits.map((v) => (
          <li key={v.session_id}>
            <Link
              href={`/visits/${v.session_id}`}
              className={`flex flex-wrap items-center justify-between gap-2 rounded-2xl border bg-white/70 px-4 py-3 shadow-sm hover:shadow-md ${
                v.risk_flag ? "border-[var(--color-danger-border)]" : "border-sage-200"
              }`}
            >
              <div>
                <p className="font-medium text-ink-900">Visit {v.session_id}</p>
                <p className="text-xs text-sage-600">
                  {new Date(v.created_at).toLocaleString()}
                </p>
              </div>
              <div className="flex items-center gap-2 text-sm text-sage-700">
                <span>PHQ-9 {v.phq9_pred?.toFixed(1) ?? "–"}</span>
                <span>HAM-D {v.hamd_pred?.toFixed(1) ?? "–"}</span>
                {v.risk_flag && (
                  <span className="rounded-full bg-[var(--color-danger)] px-2 py-0.5 text-xs font-bold text-white">
                    REFERRAL
                  </span>
                )}
              </div>
            </Link>
          </li>
        ))}
      </ul>

      <Link
        href="/patients/new"
        className="inline-block rounded-full border border-sage-300 px-4 py-2 text-sm font-medium text-sage-700 hover:bg-sage-100"
      >
        + Register a different patient
      </Link>
    </div>
  );
}

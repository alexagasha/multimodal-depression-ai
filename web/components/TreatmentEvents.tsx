"use client";

import { useEffect, useState } from "react";
import { api, ApiError, TreatmentEvent } from "@/lib/api";
import { Field, Select, TextInput, Card } from "@/components/FormField";

const EVENT_TYPE_LABEL: Record<TreatmentEvent["event_type"], string> = {
  medication_change: "Medication change",
  therapy_session: "Therapy session",
  other: "Other",
};

export default function TreatmentEvents({
  patientId,
  onChange,
}: {
  patientId: string;
  onChange?: (events: TreatmentEvent[]) => void;
}) {
  const [events, setEvents] = useState<TreatmentEvent[]>([]);
  const [eventType, setEventType] = useState<TreatmentEvent["event_type"]>("medication_change");
  const [description, setDescription] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function refresh() {
    api
      .listTreatmentEvents(patientId)
      .then((evts) => {
        setEvents(evts);
        onChange?.(evts);
      })
      .catch(() => {});
  }

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(refresh, [patientId]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!description.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.addTreatmentEvent(patientId, { event_type: eventType, description: description.trim() });
      setDescription("");
      refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card className="space-y-4">
      <div>
        <h2 className="font-display text-lg font-semibold text-sage-800">Treatment history</h2>
        <p className="text-xs text-sage-600">
          Medication and therapy changes, plotted on the severity trend above — see whether a
          change actually moved the needle at a glance, not a chart review.
        </p>
      </div>

      <ul className="space-y-2">
        {events.map((e) => (
          <li key={e.event_id} className="rounded-xl bg-sage-50 px-3 py-2">
            <div className="flex items-baseline justify-between text-xs text-sage-600">
              <span className="font-medium text-sage-800">{EVENT_TYPE_LABEL[e.event_type]}</span>
              <span>{new Date(e.event_date).toLocaleDateString()}</span>
            </div>
            <p className="mt-1 text-sm text-ink-900">{e.description}</p>
          </li>
        ))}
        {events.length === 0 && <p className="text-sm text-sage-600">No treatment events recorded.</p>}
      </ul>

      <form onSubmit={onSubmit} className="space-y-3">
        <Field label="Type">
          <Select
            value={eventType}
            onChange={(e) => setEventType(e.target.value as TreatmentEvent["event_type"])}
          >
            <option value="medication_change">Medication change</option>
            <option value="therapy_session">Therapy session</option>
            <option value="other">Other</option>
          </Select>
        </Field>
        <Field label="Description">
          <TextInput
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Started sertraline 50mg"
            required
          />
        </Field>
        {error && <p className="text-sm text-[var(--color-danger)]">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded-full bg-sage-500 px-4 py-2 text-sm font-semibold text-white hover:bg-sage-600 disabled:opacity-50"
        >
          {submitting ? "Adding…" : "Add treatment event"}
        </button>
      </form>
    </Card>
  );
}

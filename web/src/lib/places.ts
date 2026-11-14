// Places, floor plan and wayfinding types (backend: vision/navigation/places.py, api/platform.py).
export type PlaceAccess = { step_free: boolean | null; accessible_toilet: boolean | null; notes: string | null };
export type Place = {
  id: string; name: string; kind: string; position: number[]; description: string | null; aliases: string[];
  floor: string | null; access: PlaceAccess; is_start: boolean; created: string; updated?: string;
};
export type PlanTransform = { scale: number; rotation: number; x: number; y: number; opacity: number };
export type PlanInfo = {
  generated?: { file: string; width: number; height: number; px_per_unit: number; origin: number[]; units: string; url: string; points_used: number };
  uploaded?: { file: string; name: string; width: number; height: number; transform: PlanTransform; url: string } | null;
};
export type DirectionStep = { kind: "start" | "walk" | "turn" | "arrive"; text: string; notes?: string[]; length?: number; angle?: number; at?: string; to?: string };
export type Route = {
  feasible: boolean; reason?: string; profile: string; profile_label?: string; goal?: string; path: number[][]; nodes?: string[];
  length?: number; units?: string; steps: { kind: string; length: number; instruction: string }[]; warnings?: string[];
  directions?: DirectionStep[]; start_name?: string; goal_name?: string; spoken?: string;
};
export type GraphNode = { id: string; kind: "trajectory" | "facility" | "place"; pos: number[]; class_name?: string; frame?: string; confidence?: number; name?: string; place_kind?: string };
export type Graph = {
  nodes: GraphNode[]; edges: unknown[]; units: string; scale_status: string; facilities_unregistered: number; note: string;
  profiles: Record<string, string>; places: (Place & { attached_to: string; offset_from_path: number })[]; median_step: number;
};

export const PLACE_KINDS: Record<string, string> = {
  lab: "Lab", classroom: "Classroom", office: "Office", entrance: "Entrance / exit", toilet: "Toilet", lift: "Lift",
  stairs: "Stairs", stairwell: "Stairwell (landing)", ramp: "Ramp", library: "Library", canteen: "Canteen", help_desk: "Help desk",
  meeting_point: "Meeting point", other: "Other",
};

export function speak(text: string) {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return false;
  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  window.speechSynthesis.speak(u);
  return true;
}

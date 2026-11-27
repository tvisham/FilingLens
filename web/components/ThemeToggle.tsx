"use client";

import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";

function read(): Theme {
  try {
    const t = window.localStorage.getItem("theme");
    return t === "light" || t === "dark" ? t : "system";
  } catch {
    return "system";
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => setTheme(read()), []);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "system") window.localStorage.removeItem("theme");
      else window.localStorage.setItem("theme", theme);
    } catch {
      /* storage unavailable: the choice lasts for this page only */
    }
  }, [theme]);

  return (
    <label className="small" style={{ display: "flex", alignItems: "center", gap: 6, margin: 0 }}>
      Theme
      <select value={theme} onChange={(e) => setTheme(e.target.value as Theme)} style={{ width: "auto" }}>
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
  );
}

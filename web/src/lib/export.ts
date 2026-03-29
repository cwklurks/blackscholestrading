/**
 * Client-side CSV export utility.
 *
 * Builds a CSV string from headers + rows and triggers a browser download
 * via a temporary object URL. No server round-trip required.
 */

function escape(val: string | number): string {
  const s = String(val)
  return s.includes(",") || s.includes('"') || s.includes("\n")
    ? `"${s.replace(/"/g, '""')}"`
    : s
}

/**
 * Triggers a CSV file download in the browser.
 *
 * @param filename - Name of the downloaded file (e.g. "pricing-results.csv")
 * @param headers  - Column headers
 * @param rows     - Array of row arrays, each value is a string or number
 */
export function downloadCSV(
  filename: string,
  headers: string[],
  rows: (string | number)[][],
): void {
  const csvContent = [
    headers.map(escape).join(","),
    ...rows.map((row) => row.map(escape).join(",")),
  ].join("\n")

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 100)
}

import { render, screen } from "@testing-library/react"
import { describe, it, expect } from "vitest"
import KpiCard from "../src/components/KpiCard"

describe("KpiCard", () => {
  it("renders title and value", () => {
    render(<KpiCard title="Revenue" value="₹1.2Cr" subtitle="YTD" accent="teal" />)
    expect(screen.getByText("Revenue")).toBeInTheDocument()
    expect(screen.getByText("₹1.2Cr")).toBeInTheDocument()
  })
})

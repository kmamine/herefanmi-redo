import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResultCard } from "./ResultCard";

describe("ResultCard", () => {
  it("shows the verdict label and reasoning for a medical result", () => {
    render(
      <ResultCard
        result={{
          data: "Supported by peer-reviewed evidence.",
          news: "True",
          label: "Trustworthy",
          source: ["https://cdc.gov/x"],
          key: "k1",
        }}
      />,
    );
    expect(screen.getByText("Trustworthy")).toBeInTheDocument();
    expect(screen.getByText(/peer-reviewed evidence/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /cdc\.gov/ })).toHaveAttribute(
      "href",
      "https://cdc.gov/x",
    );
  });

  it("renders a neutral notice for a non-medical result (no label)", () => {
    render(
      <ResultCard
        result={{ data: "Please try to ask something related to the medical field!", key: "k2" }}
      />,
    );
    expect(screen.getByText(/medical field/)).toBeInTheDocument();
    expect(screen.queryByText("Trustworthy")).not.toBeInTheDocument();
  });

  it("renders Fake verdict styling", () => {
    render(
      <ResultCard
        result={{ data: "Unsupported miracle claim.", label: "Fake", source: [], key: "k3" }}
      />,
    );
    expect(screen.getByText("Fake")).toBeInTheDocument();
  });
});

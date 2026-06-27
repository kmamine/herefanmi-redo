import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { AdminPage } from "./AdminPage";

vi.mock("../api/client", () => ({
  api: {
    adminStats: vi.fn(),
    adminSources: vi.fn(),
    adminUsers: vi.fn(),
    adminQueries: vi.fn(),
  },
}));

describe("AdminPage", () => {
  beforeEach(() => {
    (api.adminStats as ReturnType<typeof vi.fn>).mockResolvedValue({
      users: 2,
      queries: 5,
      chunks: 12,
      per_source_chunks: { cdc: 12 },
      sources: 1,
    });
    (api.adminSources as ReturnType<typeof vi.fn>).mockResolvedValue({
      sources: [
        {
          name: "cdc",
          base_url: "https://cdc.gov",
          listing_url: "https://cdc.gov/m",
          listing_link_selector: "a",
          title_selector: "h1",
          content_selector: "article",
          date_selector: null,
          date_attr: null,
          enabled: false,
          interval_minutes: 1440,
          last_run_at: null,
          last_status: null,
        },
      ],
    });
    (api.adminUsers as ReturnType<typeof vi.fn>).mockResolvedValue({ users: [] });
    (api.adminQueries as ReturnType<typeof vi.fn>).mockResolvedValue({ queries: [] });
  });

  it("renders stats and the sources table", async () => {
    render(
      <MemoryRouter>
        <AdminPage />
      </MemoryRouter>,
    );
    expect(screen.getByRole("heading", { name: "Admin" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("cdc")).toBeInTheDocument());
    expect(screen.getByText("KB chunks")).toBeInTheDocument();
    expect(screen.getByText("Paused")).toBeInTheDocument();
  });
});

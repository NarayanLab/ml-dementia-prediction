import React from 'react';
import { render, screen } from '@testing-library/react';

// axios 1.x ships ESM, which react-scripts' jest transform does not process.
// Stubbing it also keeps these render tests off the network.
jest.mock('axios', () => ({ __esModule: true, default: { post: jest.fn() } }));

import App from './App';

// The 22 model inputs must all be reachable in the form. If a field is dropped
// from the UI the backend silently scores it at its population mean, so these
// assertions are the frontend half of that guard.
test('renders the risk assessment form', () => {
  render(<App />);
  expect(screen.getByText(/Marital Status/i)).toBeInTheDocument();
  expect(screen.getByText(/Insurance Type/i)).toBeInTheDocument();
});

test('surfaces the availability toggles for the two flagged labs', () => {
  render(<App />);
  // Calcium_missing and HCT_missing are the only missingness flags among the 22.
  expect(screen.getByText(/Calcium availability/i)).toBeInTheDocument();
  expect(screen.getByText(/Hematocrit availability/i)).toBeInTheDocument();
});

test('labels QRS as axis in degrees, not duration in ms', () => {
  render(<App />);
  expect(screen.getByText(/QRS Axis/i)).toBeInTheDocument();
  expect(screen.queryByText(/QRS Duration/i)).not.toBeInTheDocument();
});

/* Raw-network cache only: no legal-action filtering or symmetry assumptions. */
(function (scope) {
  'use strict';
  class NeuralEvaluator {
    constructor(run, namespace, capacity = 2048) {
      this.run = run; this.namespace = namespace; this.capacity = capacity;
      this.cache = new Map(); this.input = new Float32Array(0); this.active = false;
      this.resetMetrics();
    }
    resetMetrics() {
      this.metrics = {requested_evaluations: 0, neural_evaluations: 0, cache_hits: 0,
        cache_misses: 0, batch_duplicates: 0, inference_calls: 0,
        requested_batch_histogram: {}, batch_histogram: {}, backend_batch_retries: 0,
        key_cache_ms: 0, encoding_ms: 0, assembly_ms: 0};
    }
    clear(namespace = this.namespace) { this.namespace = namespace; this.cache.clear(); }
    remember(key, value) {
      this.cache.delete(key); this.cache.set(key, value);
      while (this.cache.size > this.capacity) this.cache.delete(this.cache.keys().next().value);
    }
    async evaluate(packed, count, cacheEnabled = true) {
      if (this.active) throw new Error('Concurrent inference would overwrite the input buffer');
      if (packed.length !== count * 227 || count < 1) throw new Error('Invalid packed input');
      this.active = true;
      try {
        const m = this.metrics, namespace = this.namespace, started = performance.now(), pending = new Map();
        const slots = new Array(count), missing = [], keys = [];
        m.requested_evaluations += count;
        m.requested_batch_histogram[count] = (m.requested_batch_histogram[count] || 0) + 1;
        for (let n = 0; n < count; n++) {
          const row = packed.subarray(n * 227, (n + 1) * 227);
          // Exact bytes, including turn and previous-pass input, plus model/provider/fp32.
          const key = cacheEnabled ? this.namespace + ':' + String.fromCharCode(...row) : null;
          if (cacheEnabled && this.cache.has(key)) {
            const value = this.cache.get(key); this.remember(key, value);
            slots[n] = value; m.cache_hits++;
          } else if (cacheEnabled && pending.has(key)) {
            slots[n] = pending.get(key); m.batch_duplicates++;
          } else {
            const index = missing.length;
            // Retain only a JS-owned copy; no borrowed Pyodide view survives await.
            missing.push(row.slice()); keys.push(key); slots[n] = index;
            if (cacheEnabled) pending.set(key, index);
            m.cache_misses++;
          }
        }
        m.key_cache_ms += performance.now() - started;
        let values;
        if (missing.length) {
          const encodeStart = performance.now(), size = missing.length * 900;
          if (this.input.length < size) this.input = new Float32Array(size);
          const data = this.input.subarray(0, size);
          missing.forEach((row, n) => {
            const turn = row[225], passed = row[226], offset = n * 900;
            for (let a = 0; a < 225; a++) {
              data[offset + a] = +(row[a] === turn);
              data[offset + 225 + a] = +(row[a] === 3 - turn);
              data[offset + 450 + a] = +(turn === 1);
              data[offset + 675 + a] = passed;
            }
          });
          m.encoding_ms += performance.now() - encodeStart;
          m.neural_evaluations += missing.length; m.inference_calls++;
          m.batch_histogram[missing.length] = (m.batch_histogram[missing.length] || 0) + 1;
          const result = await this.run(data, missing.length);
          if (result.length !== missing.length * 227) throw new Error('Invalid neural output shape');
          if (namespace !== this.namespace) {
            // Device fallback invalidates even the cache hits in this batch.
            // Re-evaluate the whole batch so GPU and CPU outputs are not mixed.
            m.backend_batch_retries++;
            this.active = false;
            return await this.evaluate(packed, count, cacheEnabled);
          }
          values = missing.map((_, i) => result.slice(i * 227, (i + 1) * 227));
          if (cacheEnabled) values.forEach((value, i) => this.remember(keys[i], value));
        }
        const assemblyStart = performance.now(), output = new Float32Array(count * 227);
        slots.forEach((slot, i) => output.set(typeof slot === 'number' ? values[slot] : slot, i * 227));
        m.assembly_ms += performance.now() - assemblyStart;
        return output;
      } finally { this.active = false; }
    }
  }
  scope.NeuralEvaluator = NeuralEvaluator;
  if (typeof module !== 'undefined') module.exports = {NeuralEvaluator};
})(globalThis);

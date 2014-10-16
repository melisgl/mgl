(in-package :mgl-core)

(defsection @mgl-measurer (:title "Measurers")
  "MEASURER is a part of MONITOR objects, an embedded monitor that
  computes a specific quantity (e.g. classification accuracy) from the
  arguments of event it is applied to (e.g. the model results).
  Measurers are often implemented by combining some kind of model
  specific extractor with a generic measurer function.

  All generic measurer functions return their results as multiple
  values matching the arguments of ADD-TO-COUNTER for a counter of a
  certain type (see @MGL-COUNTER) so as to make them easily used in a
  MONITOR:

      (multiple-value-call #'add-to-counter <some-counter>
                           <call-to-some-measurer>)

  The counter class compatible with the measurer this way is noted for
  each function.

  For a list of measurer functions see @MGL-CLASSIFICATION-MEASURER.")

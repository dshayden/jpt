import numpy as np



def split(z):
  """ Sample a random split proposal for UniqueBijectiveAssociation, z. """
  splits = possible_splits(z)
  if len(splits) == 0: return z, False, 0.0
  k1, t_ = splits[ np.random.choice(len(splits)) ]

  # split track k at time t inclusive: all associations <= t_ stay with to k1
  k2 = z.next_k()
  editDict_tkj = { (t, k2): j for t, j in z.to(k1).items() if t <= t_ }
  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)

  # get move log probability
  q_new_given_old = -np.log(len(splits))
  merges = possible_merges(z_)
  assert len(merges) > 0
  q_old_given_new = -np.log(len(merges))
  return z_, True, q_old_given_new - q_new_given_old

def merge(z):
  """ Sample a random merge proposal for UniqueBijectiveAssociation, z. """
  merges = possible_merges(z)
  if len(merges) == 0: return z, False, 0.0
  k1, k2 = merges[ np.random.choice(len(merges)) ]
  k_keep, k_remove = ( min(k1, k2), max(k1, k2) )

  # assign everything from k_remove to k_keep
  editDict_tkj = { (t, k_keep): j for t, j in z.to(k_remove).items() }
  z_ = z.edit(editDict_tkj, kind='tkj', inplace=False)

  # get move log probability
  splits = possible_splits(z_)
  assert len(splits) > 0
  q_old_given_new = -np.log(len(splits))
  q_new_given_old = -np.log(len(merges))

  return z_, True, q_old_given_new - q_new_given_old

def possible_merges(z):
  """ Construct all possible merges of UniqueBijectiveAssociation, z.
    
    Valid merges are all unique (k1, k2) which share no association at any time.
  """
  merges = []
  for k1 in z.ks:
    for k2 in z.ks:
      if k1 == k2: continue
      if len(np.intersect1d(z.to(k1).keys(), z.to(k2).keys())) > 0: continue
      merges.append( (k1, k2) )
  return merges

def possible_splits(z):
  """ Construct all possible splits of UniqueBijectiveAssociation, z.
    
    Valid splits are all (k, t) for which target k has an association at time t
    and at least one association at some time t' > t.
  """
  splits = []
  for k in z.ks:
    for t in list(z.to(k).keys())[:-1]: splits.append( (k, t) )
  return splits
